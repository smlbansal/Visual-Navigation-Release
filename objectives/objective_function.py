import tensorflow as tf


class Objective(object):
    def evaluate_objective(self, trajectory):
        raise NotImplementedError


class ObjectiveFunction(object):
    """
    Define an objective function.
    """
    def __init__(self):
        self.objectives = []

    def add_objective(self, objective):
        """
        Add an objective to the objective function.

        """
        self.objectives.append(objective)

    def evaluate_function_by_objective(self, trajectory):
        """
        Evaluate each objective corresponding to a system trajectory.

        """
        objective_values_by_tag = [[objective.tag, objective.evaluate_objective(trajectory)]
                                   for objective in self.objectives]
        return objective_values_by_tag

    def evaluate_function(self, trajectory):
        """
        Evaluate the entire objective function corresponding to a system trajectory.

        """
        objective_values_by_tag = self.evaluate_function_by_objective(trajectory)
        objective_function_values = 0.
        for tag, objective_values in objective_values_by_tag:
            objective_function_values += tf.reduce_mean(objective_values, axis=1)
        return objective_function_values
