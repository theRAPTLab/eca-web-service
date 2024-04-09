from aes_rl import utils
from aes_rl.args import ModelArguments
from aes_rl.model.rl import BCQ

class RL_AES(object):
    def __init__(self):
        self.providehint = self._get_trained_rl("providehint")
        self.introducecharacter = self._get_trained_rl("introducecharacter")
        self.diseasemutation = self._get_trained_rl("diseasemutation")

    def _get_trained_rl(self, aes_name):
        if aes_name in ["providehint", "introducecharacter", "diseasemutation"]:
            args = {'state_dim': 6, 'action_dim': 2, 'run_name': aes_name + '_seed1', 'eval_only': True}
        args = utils.parse_config(ModelArguments, args)
        return BCQ(args)

    def get_next_hint(self, state) -> str:
        action = self.providehint.get_action_for_state_in_np(state)

        # forcing no hint on high nlg
        if state[0] == 1:
            return "no_hint"

        if action == 0:
            return "no_hint"
        return "hint"

    def get_next_introduce_character(self, state) -> str:
        action = self.introducecharacter.get_action_for_state_in_np(state)

        if action == 0:
            return "dont_introduce"
        return "introduce"

    def get_next_disease_mutation(self, state) -> str:
        action = self.diseasemutation.get_action_for_state_in_np(state)
        if action == 0:
            return "dont_mutate"
        return "mutate"


