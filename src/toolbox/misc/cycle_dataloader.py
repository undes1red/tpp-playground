from typing import Iterable

def cycle(iterable: Iterable) -> Iterable:
    def gen_cycle():
        while True:
            for x in iterable:
                yield x
    
    return iter(gen_cycle())