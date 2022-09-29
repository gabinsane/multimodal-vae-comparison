from .mmvae_models import MOE as moe
from .mmvae_models import POE as poe
from .mmvae_models import MoPOE as mopoe
from .mmvae_models import DMVAE as dmvae
from .vae import VAE

### import contributed models
# import os
# import importlib.util
# import sys
# from glob import glob
# for file in glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "contrib/*.py")):
#     name = os.path.splitext(os.path.basename(file))[0]
#
#     spec = importlib.util.spec_from_file_location(f'./contrib/{name}', file)
#     foo = importlib.util.module_from_spec(spec)
#     sys.modules[name] = foo
#     spec.loader.exec_module(foo)
#     foo.MyClass()
#     # add package prefix to name, if required
#     module = __import__(name)
#     for member in dir(module):
#         pass
#

__all__ = [moe, poe, mopoe, dmvae, VAE]
