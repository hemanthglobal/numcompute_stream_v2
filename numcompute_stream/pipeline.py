"""
pipeline.py - Streaming-compatible Pipeline for chaining transformers and models.

Extends the original Pipeline design with partial_fit() support so that
each step can be updated incrementally on arriving data chunks.

Classes
-------
Transformer : Abstract base for streaming-compatible transformers.
Estimator   : Abstract base for streaming-compatible estimators.
Pipeline    : Chain of (name, step) pairs; supports fit, transform,
              fit_transform, partial_fit, predict.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class Transformer:
    """Abstract base for transformer steps."""

    def fit(self, X):
        return self

    def partial_fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Estimator:
    """Abstract base for estimator (model) steps."""

    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError("predict() must be implemented by subclass.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Chain transformers (and optionally a final estimator) into a single object.

    All intermediate steps must implement transform().  The final step may
    be a transformer or an estimator.  partial_fit() updates each step
    incrementally on a new chunk.

    Parameters
    ----------
    steps : list of (str, object) tuples
        Each tuple is (name, step).  Names must be unique strings and must
        not contain '__'.

    Examples
    --------
    >>> pipe = Pipeline([
    ...     ('scale', StandardScaler()),
    ...     ('model', RandomForestClassifier()),
    ... ])
    >>> pipe.partial_fit(X_chunk, y_chunk)
    >>> pipe.predict(X_test)
    """

    def __init__(self, steps):
        self._validate_steps(steps)
        self.steps = steps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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
        # All intermediate steps must have transform()
        for name, obj in steps[:-1]:
            if not hasattr(obj, "transform"):
                raise ValueError(
                    f"Intermediate step '{name}' must have a transform() method."
                )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def _intermediates(self):
        return self.steps[:-1]

    def _final_name(self):
        return self.steps[-1][0]

    def _final_step(self):
        return self.steps[-1][1]

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Fit all steps sequentially on (X, y).

        Intermediate steps are fit and transformed; the final step is fit.

        Returns
        -------
        self
        """
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
        """
        Incrementally update all steps with a new chunk (X, y).

        Each intermediate step's partial_fit() is called, then transform()
        is applied before passing data to the next step.  The final step's
        partial_fit() is called with the transformed X (and y if provided).

        Returns
        -------
        self
        """
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
        """
        Apply all transformers in sequence (no final estimator).

        Raises
        ------
        TypeError  If the final step has no transform() method.
        """
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
        """Fit then transform in one call."""
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """
        Transform through all intermediate steps, then call predict() on
        the final estimator.

        Raises
        ------
        TypeError  If the final step has no predict() method.
        """
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
        """
        Transform through intermediates then call predict_proba() on the
        final estimator.
        """
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
        """
        Return the step object with the given name.

        Raises
        ------
        KeyError  If no step with that name exists.
        """
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
