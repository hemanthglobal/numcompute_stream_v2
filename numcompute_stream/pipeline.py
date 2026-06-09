import numpy as np

class Transformer:

    def fit(self, X):
        return self

    def partial_fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class Estimator:

    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError("predict() must be implemented by subclass.")

class Pipeline:

    def __init__(self, steps):
        self._validate_steps(steps)
        self.steps = steps

    @staticmethod
    def _validate_steps(steps):
        if not steps:
            raise ValueError("Pipeline steps cannot be empty.")
        for i, step in enumerate(steps):
            if not isinstance(step, (tuple, list)) or len(step) != 2:
                raise ValueError(
                    f"Step {i} must be a (name, object) tuple, got {type(step)}."
                )
            name, obj = step
            if not isinstance(name, str):
                raise ValueError(f"Step {i} name must be a string.")
            if "__" in name:
                raise ValueError(f"Step name '{name}' cannot contain '__'.")
        for name, obj in steps[:-1]:
            if not hasattr(obj, "transform"):
                raise ValueError(
                    f"Intermediate step '{name}' must have a transform() method."
                )

    def _intermediates(self):
        return self.steps[:-1]

    def _final_name(self):
        return self.steps[-1][0]

    def _final_step(self):
        return self.steps[-1][1]

    def fit(self, X, y=None):
        X_cur = np.asarray(X, dtype=float)
        for name, step in self._intermediates():
            X_cur = step.fit(X_cur).transform(X_cur)
        final = self._final_step()
        if hasattr(final, "fit"):
            if y is not None:
                final.fit(X_cur, y)
            else:
                final.fit(X_cur)
        return self

    def partial_fit(self, X, y=None):
        X_cur = np.asarray(X, dtype=float)
        for name, step in self._intermediates():
            if hasattr(step, "partial_fit"):
                step.partial_fit(X_cur)
            X_cur = step.transform(X_cur)

        final = self._final_step()
        if hasattr(final, "partial_fit"):
            if y is not None:
                final.partial_fit(X_cur, y)
            else:
                final.partial_fit(X_cur)
        elif hasattr(final, "fit"):
            if y is not None:
                final.fit(X_cur, y)
            else:
                final.fit(X_cur)
        return self

    def transform(self, X):
        final = self._final_step()
        if not hasattr(final, "transform"):
            raise TypeError(
                f"Final step '{self._final_name()}' has no transform() method."
            )
        X_cur = np.asarray(X, dtype=float)
        for _, step in self.steps:
            X_cur = step.transform(X_cur)
        return X_cur

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        final = self._final_step()
        if not hasattr(final, "predict"):
            raise TypeError(
                f"Final step '{self._final_name()}' has no predict() method."
            )
        X_cur = np.asarray(X, dtype=float)
        for _, step in self._intermediates():
            X_cur = step.transform(X_cur)
        return final.predict(X_cur)

    def predict_proba(self, X):
        final = self._final_step()
        if not hasattr(final, "predict_proba"):
            raise TypeError(
                f"Final step '{self._final_name()}' has no predict_proba() method."
            )
        X_cur = np.asarray(X, dtype=float)
        for _, step in self._intermediates():
            X_cur = step.transform(X_cur)
        return final.predict_proba(X_cur)

    def get_step(self, name):
        for step_name, obj in self.steps:
            if step_name == name:
                return obj
        raise KeyError(
            f"No step named '{name}'. Available: {[s[0] for s in self.steps]}"
        )

    def __repr__(self):
        steps_str = ", ".join(
            f"('{n}', {o.__class__.__name__}())" for n, o in self.steps
        )
        return f"Pipeline([{steps_str}])"
