import os
import uuid

import jwt
import requests

access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")
origin = "https://api.upbit.com"


def get_balances():
    payload = {
        "access_key": access_key,
        "nonce": str(uuid.uuid4()),
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorization = "Bearer {}".format(jwt_token)
    headers = {
        "Authorization": authorization,
    }
    res = requests.get(f"{origin}/v1/accounts", headers=headers)
    return res.json()
