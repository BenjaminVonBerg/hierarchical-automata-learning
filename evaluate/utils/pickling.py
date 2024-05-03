import hashlib
import os
import pickle
import warnings
from collections.abc import MutableMapping
from functools import partial
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

base_path = Path(__file__).parent.parent.parent / "Evaluation Files"
default_path = base_path / "pickles"

def dump(path: Path | str, data, should_raise = False):
  path = Path(path)
  try:
    with open(path, "wb") as file:
      pickle.dump(data, file)
  except:
    if path.is_file():
      path.unlink(missing_ok=True)
    if should_raise:
      raise
    else:
      warnings.warn("Failed to dump pickle")


def load(path: Path | str):
  with open(path, "rb") as file:
    return pickle.load(file)

def compute_or_unpickle(path: Path, function: Callable, verbose : bool, /, *args, **kwargs):
  """
  obtains data either by:
  - computing it from a function with parameters
  - loading it from a pickle
  in the former case, the result is stored in a pickle named after the hash of the arguments
  if a file corresponding to the hash of the arguments already exists, the result is retrieved from there

  parameters:
    file_name : folder containing the pickles
    function : function for determining the operation
    args / kwargs : arguments for the function

  returns:
    function(*args, **kwargs)
  """

  # hash parameters
  parameters = (args, kwargs)
  binary_params = pickle.dumps(parameters)
  param_hash = hashlib.sha256(binary_params).hexdigest()

  # create folders
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  path = path / param_hash

  # TODO add safe version that also checks equality of arguments if hashes match
  if path.exists() and os.path.getsize(path) > 0:
    if verbose:
      print("loading from file:", path.parent.name)
    result = load(path)
  else:
    if verbose:
      print("recomputing:", path.parent.name)
    result = function(*args, **kwargs)
    dump(path, result)
  return result


T = TypeVar("T")
P = ParamSpec("P")
Fun = Callable[P, T]
Wrapper = Callable[[Fun], Fun]

class PickleCache:
  def __init__(self, base_path=".", verbose=False):
    self.base_path = Path(base_path)
    self.verbose = verbose

  def __call__(self, name=None) -> Wrapper:
    def decorator(f):
      nonlocal name
      if name is None:
        name = f.__name__

      def fun(*args, **kwargs):
        return compute_or_unpickle(self.base_path / name, f, self.verbose,  *args, **kwargs)
      return fun
    return decorator

  def add_folder(self, folder_name):
    self.base_path = self.base_path / folder_name

cached = PickleCache(default_path)

class PickledDict(MutableMapping):
  def __init__(self, folder, transform=None):
    self.folder = Path(folder)
    self.folder.mkdir(parents=True, exist_ok=True)

    self.key_transform_ = transform or str

  def key_transform(self, key):
    key = self.key_transform_(key)
    if "/" in key:
      raise RuntimeError("The key transform might point outside the folder containing the pickles")
    return self.folder / key

  def __setitem__(self, __key, __value):
    dump(self.key_transform(__key), __value)

  def __delitem__(self, __key):
    path = self.key_transform(__key)
    path.unlink(missing_ok=True)

  def __getitem__(self, __key):
    return load(self.key_transform(__key))

  def keys(self):
    return os.listdir(self.folder)

  def __len__(self):
    return len(self.keys())

  def __iter__(self):
    yield from self.keys()

  def __contains__(self, item):
    return self.key_transform(item).name in self.keys()

  def cached(self, name=None):
    def decorator(fun):
      return partial(self.get_or_compute, name or fun.__name__, fun)
    return decorator

  def get_or_compute(self, key, compute):
    if key not in self:
      ret = compute()
      self[key] = ret
      return ret
    return self[key]