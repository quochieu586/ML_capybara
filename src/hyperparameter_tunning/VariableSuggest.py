from abc import ABC, abstractmethod
from typing import List
import optuna

class VariableSuggestion(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_suggest(self, trial: optuna.trial.Trial):
        pass

    @abstractmethod
    def get_copy(self, name: str):
        """
            Return new copy of this suggestion with different name. This is used to tun list hyperparameters.
        """
        pass

class IntVariableSuggestion(VariableSuggestion):
    """
        Suggest int value for optuna model
    """
    def __init__(self, name: str, start: int, end: int):
        super().__init__(name)

        self.start = start
        self.end = end

    def get_suggest(self, trial: optuna.trial.Trial):
        return trial.suggest_int(self.name, self.start, self.end)
    
    def get_copy(self, name: str):
        return IntVariableSuggestion(name, self.start, self.end)

    
class FloatVariableSuggestion(VariableSuggestion):
    """
        Suggest float value for optuna model
    """
    def __init__(self, name: str, start: float, end: float):
        super().__init__(name)

        self.start = start
        self.end = end

    def get_suggest(self, trial: optuna.trial.Trial):
        return trial.suggest_float(self.name, self.start, self.end)
    
    def get_copy(self, name: str):
        return FloatVariableSuggestion(name, self.start, self.end)



class CategoricalVariableSuggestion(VariableSuggestion):
    def __init__(self, name: str, classes: list):
        super().__init__(name)

        self.classes = classes

    def get_suggest(self, trial: optuna.trial.Trial):
        return trial.suggest_categorical(self.name, self.classes)
    
    def get_copy(self, name):
        return CategoricalVariableSuggestion(name, self.classes)
    

    
class ListVariableSuggestion(VariableSuggestion):
    """
        This Concrete class use for suggest the variable-length hyperparameter 
    """
    def __init__(self, name: str, min_len: int, max_len: int, member: VariableSuggestion):
        super().__init__(name)

        self.min_len = min_len
        self.max_len = max_len
        self.member = member

    def get_suggest(self, trial: optuna.trial.Trial):
        """
            Tunning size of hyperparameters and each element of it
        """
        size = trial.suggest_int(f"len_{self.name}", self.min_len, self.max_len)
        suggestions = [self.member.get_copy(f"{self.name}_{idx}") for idx in range(size)]

        return [suggestion.get_suggest(trial) for suggestion in suggestions]

    def get_copy(self, name: str):
        return ListVariableSuggestion(self.name, self.min_len, self.max_len, self.member)