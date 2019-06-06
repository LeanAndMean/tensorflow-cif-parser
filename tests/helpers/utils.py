# -*- coding: utf-8 -*-
import functools
from pyinstrument import Profiler


def profile_with_pytinstrument(func):
    """Decorator for wrapping functions with a lightweight profiler."""
    @functools.wraps(func)
    def pytinstrument_decorator(*args, **kwargs):
        # Create profiler instance before function is called.
        profiler = Profiler()
        profiler.start()
        # Call function.
        func_return = func(*args, **kwargs)
        # Stop profiler and print results after function is called.
        profiler.stop()
        print(profiler.output_text(unicode=False, color=True))
        return func_return
    return pytinstrument_decorator
