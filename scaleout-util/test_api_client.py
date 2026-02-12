"""Quick test script for APIClient with TokenManager integration."""

from scaleoututil.api.client import APIClient


if __name__ == "__main__":

    client = APIClient(host="http://edge.scaleout.com", verify=False, token="ory_rt_zz3UR9iNLBm_fuTgxfYzLIfqwcLvnl_8I-8oV_trb4o.aDFUW2MPjV7aV5Hu7zENqZ0wjLdFli5i2iZez24sCz4")

    model = client.get_active_model()

    print("Active model ID:", model)