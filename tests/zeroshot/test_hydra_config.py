import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.codex import _ChatOpenAICodex

from zeroshot.models import SGLangChatOpenAI
from zeroshot.pipeline.event_logging import ConsoleReporter
from zeroshot.pipeline.messages import MessageBuilder, PromptTemplate
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.workflow import (
    SemanticHypothesis,
    create_agent,
    create_reconstruction_graph,
)

CONFIG_DIR = Path(__file__).parents[2] / "zeroshot" / "configs"


def test_default_config_instantiates_message_builder() -> None:
    config_path = CONFIG_DIR / "default.yaml"
    assert config_path.is_file(), config_path

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(config_name="default")

    builder = instantiate(config.message_builder)
    console_reporter = instantiate(config.console)
    assert isinstance(builder, MessageBuilder)
    assert isinstance(console_reporter, ConsoleReporter)


def test_gemma4_ollama_config_instantiates_chat_openai() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=gemma4_ollama"],
        )

    model = instantiate(config.model)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gemma4:e2b"
    assert model.openai_api_base == "http://127.0.0.1:11434/v1"
    assert model.request_timeout == 600.0
    assert model.max_retries == 0


def test_qwen3_6_sglang_config_instantiates_chat_openai() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=qwen3_6_sglang"],
        )

    model = instantiate(config.model)

    assert isinstance(model, SGLangChatOpenAI)
    assert model.model_name == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert model.openai_api_base == "http://127.0.0.1:30000/v1"
    assert model.request_timeout == 600.0
    assert model.max_retries == 0
    assert model.streaming is True
    assert model.stream_usage is True
    assert model.max_tokens == 65536
    assert model.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_qwen3_6_sglang_thinking_and_output_limit_are_overridable() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=[
                "model=qwen3_6_sglang",
                "model.extra_body.chat_template_kwargs.enable_thinking=false",
                "model.max_tokens=1024",
            ],
        )

    model = instantiate(config.model)

    assert model.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert model.max_tokens == 1024


def test_gpt5_6_luna_codex_config_instantiates_oauth_model() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=gpt5_6_luna_codex"],
        )

    model = instantiate(config.model)

    assert isinstance(model, _ChatOpenAICodex)
    assert model.model_name == "gpt-5.6-luna"
    assert model.output_version == "responses/v1"
    assert model.streaming is True
    assert model.use_responses_api is True
    assert model.store is False
    assert model.openai_api_base == "https://chatgpt.com/backend-api/codex"
    assert model.request_timeout == 600.0
    assert model.max_tokens is None
    # A sweep over a remote endpoint loses whole samples to transient resets.
    assert model.max_retries >= 1


def test_a_sweep_only_has_to_override_the_sample_id() -> None:
    """Every input path derives from `sample_id`, so a sweep sets that alone."""

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["sample.sample_id=000405"],
        )

    assert config.sample.sample_id == "000405"
    assert config.sample.dxf_path.endswith("/000405.dxf")
    assert config.sample.target_step_path.endswith("/000405.step")
    assert all(
        path.endswith("/000405.png") for path in config.sample.render3d_paths.values()
    )


def test_the_sample_id_keeps_its_leading_zeros() -> None:
    """`000405` must stay a string: as an int it would interpolate as `405`."""

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["sample.sample_id=000405"],
        )

    assert isinstance(config.sample.sample_id, str)


def test_hydra_writes_its_job_output_beside_the_sample_artifacts() -> None:
    """A sample directory has to say what produced it.

    Hydra's default dated tree has no link back to `artifact_root`, so the
    resolved config would be unfindable from the artifacts it explains.
    """

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["sample.sample_id=000405", "artifact_root=outputs/baseline"],
            return_hydra_config=True,
        )

    assert config.hydra.run.dir == "outputs/baseline/000405"
    assert config.hydra.sweep.dir == "outputs/baseline"
    assert config.hydra.sweep.subdir == "000405"


def test_reruns_fail_by_default() -> None:
    """`skip` silently keeps prior results, so it must be asked for."""

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(config_name="default")

    assert config.on_existing == "fail"


def test_the_workflow_is_a_selectable_group_carrying_its_own_settings() -> None:
    """Graph changes ship as a new config option, not as an edit to graph.py.

    Each run then records which graph it used in its own `.hydra/config.yaml`,
    which is what makes two runs comparable after the fact.
    """

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=[
                "workflow=baseline",
                # Overridden rather than read, so flipping a checked-in default
                # is an experiment, not a test failure.
                "workflow.coder.max_turns=5",
                "workflow.coder.announce_turn_budget=true",
                "workflow.model_retries=1",
            ],
        )

    graph_factory = instantiate(config.workflow)
    assert graph_factory.func is create_reconstruction_graph
    # One agent is one block: a config names who it is, which model answers for
    # it and how long it may run, and leaves the tools to the graph.
    assert set(graph_factory.keywords) == {
        "semantic_hypothesizer",
        "semantic_reviewer",
        "coder",
        "model_retries",
    }
    assert graph_factory.keywords["model_retries"] == 1

    coder = graph_factory.keywords["coder"]
    assert coder.func is create_agent
    assert coder.keywords["role"] == "coder"
    assert PromptTemplate(coder.keywords["role"]).path.is_file()
    assert coder.keywords["max_turns"] == 5
    assert coder.keywords["announce_turn_budget"] is True
    # `model: ${model}` follows the run, so one override still swaps every
    # agent that did not ask for a backend of its own.
    assert isinstance(coder.keywords["model"], BaseChatModel)
    assert coder.keywords["model"].model_name == "gemma4:e2b"

    # A reasoning stage owes a typed answer, and which strategy produces it is
    # the config's call rather than something read off the model's profile.
    hypothesizer = graph_factory.keywords["semantic_hypothesizer"]
    assert isinstance(hypothesizer.keywords["response_format"], ProviderStrategy)
    assert hypothesizer.keywords["response_format"].schema is SemanticHypothesis
    assert "response_format" not in coder.keywords
    assert "agent" not in config


def test_every_rerun_policy_the_config_documents_is_accepted() -> None:
    """A policy named only in a comment is one a sweep discovers at runtime."""

    for policy in ("fail", "skip", "retry"):
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR.resolve()),
            version_base="1.3",
        ):
            config = compose(config_name="default", overrides=[f"on_existing={policy}"])
        assert config.on_existing == policy
        PipelineRunner(
            sandbox_runner=SandboxRunner(
                python_executable=Path(sys.executable), default_timeout_s=30
            ),
            graph_factory=instantiate(config.workflow),
            message_builder=instantiate(config.message_builder),
            artifact_root="unused",
            renderer=instantiate(config.renderer),
            on_existing=config.on_existing,
        )
