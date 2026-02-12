import os
import random  # noqa: F401
from typing import Optional, Dict, List, Tuple  # noqa: F401

import numpy as np  # noqa: F401

from scaleoututil.api.client import APIClient
from scaleoututil.logging import ScaleoutLogger  # noqa: F401
from scaleoututil.serverfunctions.serverfunctionsbase import ServerFunctionsBase, RoundType  # noqa: F401


if os.getenv("REDUCER_SERVICE_HOST") and os.getenv("REDUCER_SERVICE_PORT"):
    host = f"{os.getenv('REDUCER_SERVICE_HOST')}:{os.getenv('REDUCER_SERVICE_PORT')}/internal"
    port = None
else:
    host = "scaleout-api-server"
    port = 8092

api_client = APIClient(host=host, port=port)
print = ScaleoutLogger().info


# --- Combiner context ---
_COMBINER_NAME: Optional[str] = None


# combiner id can be useful e.g. for sharing attributes across sessions and combiners using the api client
def get_combiner_name() -> str:
    """Return the ID of the current combiner.

    The combiner ID is injected by the Scaleout runtime and is only
    available while code is executing in a combiner context. It can be
    used, for example, together with :data:`api_client` to share
    attributes or state across sessions and combiners.

    Returns:
        str: The identifier of the current combiner.

    Raises:
        RuntimeError: If the combiner ID has not been set yet, for
            example when called outside of a combiner context or before
            the runtime has initialised it.

    """
    if _COMBINER_NAME is None:
        raise RuntimeError("combiner_name not set.")
    return _COMBINER_NAME
