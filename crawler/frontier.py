from collections import deque

class Frontier:

    def __init__(self, seeds, max_depth=2):
        self.queue = deque()
        self.visited = set()

        for url in seeds:
            self.queue.append((url, 0))

        self.max_depth = max_depth

    def add(self, url, depth):

        if url not in self.visited and depth <= self.max_depth:
            self.queue.append((url, depth))

    def next(self):

        if self.queue:
            url, depth = self.queue.popleft()
            self.visited.add(url)
            return url, depth

        return None, None

    def has_next(self):
        return len(self.queue) > 0