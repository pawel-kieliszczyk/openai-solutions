import random
import sum_tree


class PrioritizedReplayMemory(object):
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = sum_tree.SumTree(capacity)

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
