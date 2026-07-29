"""Familiars Tools. EvolTrainer.

Copyright (c) 2026 IgorNk500"""

import torch
import numpy as np
import os
import copy
from typing import Optional
from transformers import Trainer, TrainingArguments
from transformers.utils.import_utils import requires
#from transformers.trainer_utils import EvalPrediction

from .manager import EvolManager

@requires(
    backends=(
        "torch",
        "accelerate",
    )
)
class EvolTrainer(Trainer):
    """**Evolution trainer for familiars models inheriting transformers.Trainer.**

    Instead of gradients, it uses a genetic algorithm over a population of models.

    **!!!WARNING!!! The trainer was created only for familiars models. Use with other models can have unpredictable consequences. !!!WARNING!!!**"""

    def __init__(
        self,
        mgr: EvolManager,
        args: TrainingArguments,
        mutation_rate: float = 0.1,
        mutation_power: float = 0.02,
        elitism: int = 5,
        episodes_per_net: int = 3,
        **kwargs
    ):
        model = mgr.model
        super().__init__(
            model=model,
            args=args,
            train_dataset=None,  # We don't use train dataset
            **kwargs)
        mgr.actions.trigger("start")
        self.mgr = mgr
        self.population_size = mgr.population_size
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.elitism = elitism
        self.episodes_per_net = episodes_per_net

        mgr.actions.trigger("pre_init")

        # Creating the initial population: we save the state_dict of each individual
        base_state = copy.deepcopy(model.state_dict())
        self.population = [copy.deepcopy(base_state) for _ in range(self.population_size)]
        # The best individual and its fitness
        self.best_state = base_state
        self.best_fitness = -np.inf

        # Disable gradients, they are not needed
        self.model.eval()

        mgr.actions.trigger("post_init")

    def train(self, resume_from_checkpoint: Optional[str] = None, **kwargs):
        """Main evol training entry point"""
        # The standard start of train – triggers on_train_begin callbacks
        self.callback_handler.train_begin(self.args, self.state, self.control)

        stop_action = self.mgr.actions.stop_train_event
        trg = self.mgr.actions.trigger

        trg("pre_cycle")
        # Play->Analyse->Train cycle
        gen = 0
        while not (stop_action and stop_action.is_set()):
            gen += 1
            self.state.epoch = gen
            self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            trg("pre_using")
            # STEP 1: PLAY GAME ALL INDIVIDUAL
            fitness_scores = []
            for idx, individual_state in enumerate(self.population):
                if stop_action and stop_action.is_set():
                    break

                # Load model weights
                self.model.load_state_dict(individual_state)
                self.model.eval()

                # Play
                reward = self._play_cycle()
                fitness_scores.append(reward)

                self.log({f"individual_{idx}_fitness": reward})

            if stop_action and stop_action.is_set():
                break

            trg("post_using")

            trg("pre_train")
            # STEP 2: UPDATE BEST INDIVIDUAL
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_state = copy.deepcopy(self.population[best_idx])
            # Set best model in the manager
            self.mgr.best_state = self.best_state

            # Log
            logs = {
                "mean_fitness": float(np.mean(fitness_scores)),
                "max_fitness": float(np.max(fitness_scores)),
                "best_fitness_ever": self.best_fitness
            }
            self.log(logs)

            # STEP 3: EVALUATE
            self._evolve_population(fitness_scores)

            self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            trg("post_train")
        trg("ending")


    def _play_cycle(self) -> int:
        """**Game cycle.** *(1 life)* Works with ScreenIO or FamiliarIO; GameActions; EvolManager."""
        io = self.mgr.io

        dead = False
        while not dead:
            pass


    # ------------------------------------------------------------------
    # Evaluation best model
    # ------------------------------------------------------------------
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval", episodes: int = 10):
        self.model.load_state_dict(self.best_state)
        total_reward = 0.0
        with torch.no_grad():
            for _ in range(episodes):
                state = self.env.reset()
                done = False
                while not done:
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'state_dim'):
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                    elif hasattr(self.model, 'config') and hasattr(self.model.config, 'vit_name') is not None:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    else:
                        state_tensor = state
                    action_logits, args = self.model(state_tensor)
                    action = torch.argmax(action_logits, dim=1).item()
                    args_np = args.squeeze(0).cpu().numpy()
                    next_state, reward, done, _ = self.env.step((action, args_np))
                    total_reward += reward
                    state = next_state
        avg_reward = total_reward / episodes
        metrics = {f"{metric_key_prefix}_mean_reward": avg_reward}
        self.log(metrics)
        return metrics



    # ------------------------------------------------------------------
    # Genetic operations
    # ------------------------------------------------------------------
    def _evolve_population(self, fitness_scores):
        # Save best
        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_population = [copy.deepcopy(self.population[i]) for i in sorted_indices[:self.elitism]]

        # Probabilities of parents' choice (softmax on fitness)
        probs = torch.softmax(torch.tensor(fitness_scores, dtype=torch.float), dim=0).numpy()

        while len(new_population) < self.population_size:
            parent1_idx = np.random.choice(len(self.population), p=probs)
            parent2_idx = np.random.choice(len(self.population), p=probs)
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)

        self.population = new_population

    def _crossover(self, state1, state2):
        child = copy.deepcopy(state1)
        with torch.no_grad():
            for key in child.keys():
                if state1[key].dtype in (torch.float32, torch.float64, torch.bfloat16):  # Only float
                    mask = torch.rand_like(state1[key]) < 0.5
                    child[key] = torch.where(mask, state1[key], state2[key])
        return child

    def _mutate(self, state):
        with torch.no_grad():
            for key in state.keys():
                if state[key].dtype in (torch.float32, torch.float64):
                    mask = torch.rand_like(state[key]) < self.mutation_rate
                    noise = torch.randn_like(state[key]) * self.mutation_power
                    state[key] += mask.float() * noise

    # ------------------------------------------------------------------
    # Saving / Loading population
    # ------------------------------------------------------------------
    def save_state(self):
        """The standard Trainer method calls save_state() to save the state of the optimizer, etc.
        We are additionally preserving the population."""
        super().save_state()
        # Save population
        population_path = os.path.join(self.args.output_dir, "population.bin")
        torch.save({
            "population": self.population,
            "best_state": self.best_state,
            "best_fitness": self.best_fitness
        }, population_path)

    def load_state(self, checkpoint_folder):
        """Load state from checkpoint"""
        #super().load_state(checkpoint_folder)
        population_path = os.path.join(checkpoint_folder, "population.bin")
        if os.path.exists(population_path):
            checkpoint = torch.load(population_path)
            self.population = checkpoint["population"]
            self.best_state = checkpoint["best_state"]
            self.best_fitness = checkpoint["best_fitness"]
            self.model.load_state_dict(self.best_state)
