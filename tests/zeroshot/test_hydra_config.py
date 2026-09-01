import sys
from inspect import signature
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from langchain_openrouter import ChatOpenRouter

from zeroshot.models import SGLangChatOpenAI
from zeroshot.pipeline.event_logging import ConsoleReporter
from zeroshot.pipeline.messages import ArtifactPresenter, PromptTemplate
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.workflow import create_agent, create_reconstruction_graph

CONFIG_DIR = Path(__file__).parents[2] / "zeroshot" / "configs"


def test_default_config_instantiates_artifact_presenter() -> None:
    config_path = CONFIG_DIR / "default.yaml"
    assert config_path.is_file(), config_path

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(config_name="default")

    presenter = instantiate(config.artifact_presenter)
    console_reporter = instantiate(config.console)
    assert isinstance(presenter, ArtifactPresenter)
    assert isinstance(console_reporter, ConsoleReporter)
    assert console_reporter._muted_graph_nodes == frozenset()
    assert config.resume_from is None


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


@pytest.mark.parametrize(
    ("model_config", "model_name"),
    [
        ("gpt5.6_luna_codex", "gpt-5.6-luna"),
        ("gpt5.6_terra_codex", "gpt-5.6-terra"),
    ],
)
def test_gpt5_6_codex_config_instantiates_oauth_model(
    model_config: str, model_name: str
) -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=[f"model={model_config}"],
        )

    model = instantiate(config.model)

    assert isinstance(model, _ChatOpenAICodex)
    assert model.model_name == model_name
    assert model.output_version == "responses/v1"
    assert model.streaming is True
    assert model.use_responses_api is True
    assert model.store is False
    assert model.openai_api_base == "https://chatgpt.com/backend-api/codex"
    assert model.request_timeout == 600.0
    assert model.max_tokens is None
    # Agent middleware is the sole retry owner; SDK-level retries stay off.
    assert model.max_retries == 0


def test_glm5_3_flash_openrouter_config_instantiates_chat_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=glm5.3_flash_openrouter"],
        )

    model = instantiate(config.model)

    assert isinstance(model, ChatOpenRouter)
    assert model.model_name == "z-ai/glm-5.3-flash"
    assert model.request_timeout == 600000
    assert model.max_retries == 0


@pytest.mark.parametrize(
    ("model_config", "strategy"),
    [
        ("gemma4_ollama", "provider"),
        ("glm5.3_flash_openrouter", "tool"),
    ],
)
def test_the_backend_chosen_decides_how_every_agent_is_asked_for_structured_output(
    model_config: str,
    strategy: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend that takes no json_schema response format has to bind the
    answer as a tool instead, so the strategy follows the `model` group rather
    than being restated per agent."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=[f"model={model_config}"],
        )

    builders = [
        "semantics_agent_builder",
        "operations_agent_builder",
        "coding_agent_builder",
        "audit_agent_builder",
    ]
    assert [
        config.workflow[builder].response_format_strategy for builder in builders
    ] == [strategy] * len(builders)


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
                "workflow=staged",
                # Overridden rather than read, so flipping a checked-in default
                # is an experiment, not a test failure.
                "workflow.coding_agent_builder.max_turns=5",
                "workflow.coding_agent_builder.announce_turns=true",
                "workflow.coding_agent_builder.model_retries=1",
            ],
        )

    graph_factory = instantiate(config.workflow)
    assert graph_factory.func is create_reconstruction_graph
    # A stage is one block, an agent is one block inside it: a config names who
    # answers, on which model, for how long, and leaves the tools and the
    # contracts to the code.
    assert set(graph_factory.keywords) == {
        "semantics_agent_builder",
        "operations_agent_builder",
        "coding_agent_builder",
        "audit_agent_builder",
        "max_audit_reject_count",
        "max_stage_validation_retries",
    }

    coder = graph_factory.keywords["coding_agent_builder"]
    assert coder.func is create_agent
    assert coder.keywords["role"] == "coder"
    assert PromptTemplate(f"roles/{coder.keywords['role']}").path.is_file()
    assert coder.keywords["max_turns"] == 5
    assert coder.keywords["announce_turns"] is True
    assert coder.keywords["model_retries"] == 1
    # `model: ${model}` follows the run, so one override still swaps every
    # agent that did not ask for a backend of its own.
    coder_model = coder.keywords["model"]
    assert isinstance(coder_model, ChatOpenAI)
    assert coder_model.model_name == "gemma4:e2b"

    stage = graph_factory.keywords["semantics_agent_builder"]
    assert stage.func is create_agent
    assert stage.keywords["role"] == "semantic_hypothesizer"
    assert PromptTemplate("roles/semantic_hypothesizer").path.is_file()
    assert stage.keywords["response_format_strategy"] == "provider"
    assert stage.keywords["model"].model_name == "gemma4:e2b"

    operations = graph_factory.keywords["operations_agent_builder"]
    assert operations.func is create_agent
    assert operations.keywords["role"] == "operation_planner"
    assert PromptTemplate("roles/operation_planner").path.is_file()
    assert "output_schema" not in stage.keywords
    assert "agent" not in config


def test_the_continued_workflow_runs_the_reasoning_stages_as_one_agent() -> None:
    """The variant is the staged graph with the thread shared, so it has to
    inherit staged's settings rather than restate them, and it has to give the
    three stages that share the thread one role."""
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(config_name="default", overrides=["workflow=continued"])

    graph_factory = instantiate(config.workflow)

    assert graph_factory.func is create_reconstruction_graph
    assert graph_factory.keywords["share_thread"] is True

    roles = {
        graph_factory.keywords["semantics_agent_builder"].keywords["role"],
        graph_factory.keywords["operations_agent_builder"].keywords["role"],
        graph_factory.keywords["coding_agent_builder"].keywords["role"],
    }
    assert roles == {"cad_reconstructor"}
    assert PromptTemplate("roles/cad_reconstructor").path.is_file()

    # Hydra partials accept unknown keywords and otherwise defer this failure
    # until the first sample builds its graph. Check every configured builder
    # against its callable now so configuration drift fails in this test.
    for key in (
        "semantics_agent_builder",
        "operations_agent_builder",
        "coding_agent_builder",
        "audit_agent_builder",
    ):
        builder = graph_factory.keywords[key]
        signature(builder.func).bind_partial(**builder.keywords)

    # The audit reads the result on its own, so it keeps its own role.
    assert (
        graph_factory.keywords["audit_agent_builder"].keywords["role"]
        == "output_auditor"
    )
    # Inherited from staged rather than restated here.
    assert "max_audit_reject_count" in graph_factory.keywords


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
            artifact_presenter=instantiate(config.artifact_presenter),
            artifact_root="unused",
            renderer=instantiate(config.renderer),
            on_existing=config.on_existing,
        )
