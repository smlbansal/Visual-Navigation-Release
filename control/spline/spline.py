class Spline:

  def fit(self, start, goal):
    raise NotImplementedError

  def evaluate(self, ts):
    raise NotImplementedError

  def render(self, ax):
    raise NotImplementedError
