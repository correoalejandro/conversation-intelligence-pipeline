"""
SAS Helper
Handles CAS uploads, MAS calls, token refresh.
"""

import base64
import json
import time
import os
from json import loads, dumps

# import certifi
import requests
import swat

from sasctl import Session
from sasctl.services import microanalytic_score as mas

# certifi.where()

# import os
swat.options.cas.print_messages = True

# set/create variables
client_id = "api.client"
client_secret = "api.secret"
#Jose.Poveda-03-03-2025: Se cambia el server_name de ssemonthly a innovationlab
sas_server_name = "create.demo.sas.com"
# sas_server_name = "ssemonthly.demo.sas.com"

baseURL = f"https://{sas_server_name}"
sas_env = sas_server_name.split(".")[0]

# cas_server_name = "cas-shared-default-http"
cas_server_name = "cas-shared-epic-http"


# enccode client string
client_string = client_id + ":" + client_secret
message_bytes = client_string.encode("ascii")
base64_bytes = base64.b64encode(message_bytes)
base64_message = base64_bytes.decode("ascii")

# copy resfresh token from txtfile
basepath = "."
share_data = basepath + "/0-codigos_conexiones/"
USER_ID = "aleja"  # 👈 tu alias personalizado

file_access_token = os.path.join(share_data, f"access_token_{sas_env}_{USER_ID}.txt")
file_refresh_token = os.path.join(share_data, f"refresh_token_{sas_env}_{USER_ID}.txt")



def refreshToken():
    file = open(file_refresh_token)
    # read the file as a list
    refresh_token = file.readlines()
    # close the file
    file.close()
    # print(file_refresh_token)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    # changed from 'params=payload' to 'data=payload'
    response = requests.post(
        f"{baseURL}/SASLogon/oauth/token", headers=headers, data=payload
    )
    result = False
    if response.status_code == 200:
        access_token = json.loads(response.text)["access_token"]
        refresh_token = json.loads(response.text)["refresh_token"]
        print(json.dumps(response.json(), indent=4, sort_keys=True))

        # Create access_token.txt file
        with open(file_access_token, "w") as f:
            f.write(access_token)
        # print("El token de acceso se almacenó para usted como" + file_access_token)
        # Create access_token.txt file
        with open(file_refresh_token, "w") as f:
            f.write(refresh_token)
        # print("El token de acceso se almacenó para usted como" + file_access_token)
        result = True
    else:
        result = False

    return result


def httpConnSwat():
    # copy access token from txtfile
    file = open(file_access_token)
    # read the file as a list
    data = file.readlines()
    # close the file
    file.close()
    access_token = data[0]
    # # Nombre del archivo .pem
    file_pem = "ssemonthly-rootCA-Intermidiates_4CLI.pem"
    # # Archivo .pem certificado SSL
    cert = "./0-codigos_conexiones/cert/"
    ssl_ca_list = cert + file_pem
    # ssl_ca_list = "https://sww.sas.com/~micarm/SSEMonthly-tid-bits/ssemonthly-rootCA-Intermidiates_4CLI.pem"

    # Establecer conexión a CAS a través de SWAT
    # logging.debug("***httpconn()***")

    intentos = 3  # Número de intentos máximos permitidos

    while intentos > 0:
        try:
            # logging.debug("[[******{0}******]]".format(intentos))
            connected = swat.CAS(
                f"{baseURL}/{cas_server_name}",
                username=None,
                password=access_token,
                ssl_ca_list=ssl_ca_list,
                protocol="https",
                name="0_0_main_uoc_adam"
            )
            connected.setSessOpt(timeout=7500)
            break
        except Exception as e:
            # logging.critical(f"***Ocurrió un error: {e}")
            print(f"***Ocurrió un error: {e}")
            rtoken = refreshToken()
            print(f"*** [[{rtoken}]]")
            # logging.info(rtoken)
            time.sleep(1)
            intentos -= 1
            # logging.warning(f"***Te quedan {intentos} intentos")

    return connected


def list_modules_sas():
    conn = httpConnSwat()

    file = open(file_access_token)
    # read the file as a list
    access_token = file.readlines()
    # close the file
    file.close()
    # print(access_token)

    sess = Session(sas_server_name, token=access_token[0],
                   protocol="https", port=443)
    print(sess)
    print("\n")
    # Con la libreria SASCTL, listamos los despliegues realizados al componente MAS
    list = mas.list_modules()
    # loads(dumps(list))
    # type(list)

    name = "_001_"

    for i in range(0, len(list)):
        module = (list[i]['links'][1]['href'])
        if name in module:
            print(i, module)
    print("\n")
    # Recuperamos el modulo xgboost_hmeq, y revisamos su información
    module = mas.get_module(
        "mm_uoc_001_v1_svm")  # este funciona ok
    # module = mas.get_module("xgboost_hmeq")
    print(loads(dumps(module)))
    print("\n")
    step = mas.get_module_step(module, 'score')
    print(loads(dumps(step)))
    conn.terminate()

    return True
