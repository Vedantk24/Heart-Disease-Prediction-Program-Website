import requests

url = "http://localhost:5000/predict"
r = requests.post(url,json={"Age":45, 
                            "Gender":1,
                            "CP":3 ,
                            "RBP":120, 
                            "SC":110, 
                            "FBS":1,
                            "RER":1, 
                            "MHR":130, 
                            "EIA":1 ,
                            "ST":26,
                            "SST":1,
                            "Flourosopy":3,
                            "Thal":1})

print(r.json())