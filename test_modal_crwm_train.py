from __future__ import annotations

import unittest

import modal_crwm_train


class ModalCrwmTrainTests(unittest.TestCase):
    def test_invoke_remote_runner_uses_spawn_then_get(self) -> None:
        events: list[tuple[str, object, object] | tuple[str]] = []
        expected_result = {"train_loss": 1.23}
        config = {"device": "cuda"}
        contract = {"robot": {}}

        class FakeFunctionCall:
            def get(self) -> dict[str, float]:
                events.append(("get",))
                return expected_result

        class FakeRemoteRunner:
            def spawn(
                self,
                submitted_config: dict[str, object],
                submitted_contract: dict[str, object],
            ) -> FakeFunctionCall:
                events.append(("spawn", submitted_config, submitted_contract))
                return FakeFunctionCall()

        result = modal_crwm_train._invoke_remote_runner(FakeRemoteRunner(), config, contract)

        self.assertEqual(result, expected_result)
        self.assertEqual(events, [("spawn", config, contract), ("get",)])

    def test_wandb_enabled_detects_config_toggle(self) -> None:
        self.assertFalse(modal_crwm_train._wandb_enabled({}))
        self.assertFalse(modal_crwm_train._wandb_enabled({"wandb": {"enabled": False}}))
        self.assertTrue(modal_crwm_train._wandb_enabled({"wandb": {"enabled": True}}))

    def test_select_remote_runner_uses_wandb_function_when_enabled(self) -> None:
        self.assertIs(
            modal_crwm_train._select_remote_runner({"wandb": {"enabled": False}}),
            modal_crwm_train.run_training,
        )
        self.assertIs(
            modal_crwm_train._select_remote_runner({"wandb": {"enabled": True}}),
            modal_crwm_train.run_training_wandb,
        )


if __name__ == "__main__":
    unittest.main()
