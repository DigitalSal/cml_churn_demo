import requests
import json
import logging


class CMLApi:
    """This classs is a wrapper for calls to the internal CML api

    Attributes: 
        host (str): URL for the CML instance host. 
        username (str): Current username.
        api_key (str): API key.
        project_name (str): Project name.
    """

    def __init__(self, host, username, api_key, project_name, log_level=logging.INFO):
        self.host = host
        self.username = username
        self.api_key = api_key
        self.project_name = project_name
        logging.basicConfig(level=log_level)

        logging.debug("Api Initiated")

    def default_engine(self, params):
        """Get the default engine for the given project

        Arguments:
            params {dict} -- None needed actually.

        Returns:
            dict -- [dictionary containing default engine details]
        """
        get_engines_endpoint = "/".join([self.host, "api/v1/projects",
                                         self.username, self.project_name, "engine-images"])

        res = requests.get(
            get_engines_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("User details retrieved")

        return response

    def get_user(self, params):

        get_user_endpoint = "/".join([self.host, "api/v1/users",
                                      self.username])
        res = requests.get(
            get_user_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("User details retrieved")

        return response

    def get_project(self, params):
        get_project_endpoint = "/".join([self.host, "api/v1/projects",
                                         self.username, self.project_name])
        res = requests.get(
            get_project_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("Project details retrieved")

        return response

    def get_jobs(self, params):
        create_job_endpoint = "/".join([self.host, "api/v1/projects",
                                        self.username, self.project_name, "jobs"])
        res = requests.get(
            create_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 201):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("List of jobs retrieved")

        return response

    def create_job(self, params):
        create_job_endpoint = "/".join([self.host, "api/v1/projects",
                                        self.username, self.project_name, "jobs"])
        res = requests.post(
            create_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 201):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("Job created")
        return response
      
    def create_environment_variable(self, params):
        create_job_endpoint = "/".join([self.host, "api/v1/projects",
                                        self.username, self.project_name, "environment"])
        res = requests.put(
            create_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )
        #response = res.json()
        if (res.status_code != 204):
            logging.error("Repons code was " + res.status_code)
        else:
            logging.debug("Environment variable created")
        return res.status_code

    def start_job(self, job_id, params):
        start_job_endpoint = "/".join([self.host, "api/v1/projects",
                                       self.username, self.project_name, "jobs", str(job_id), "start"])
        res = requests.post(
            start_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.text
        if (res.status_code != 201):
            logging.error(response)
            logging.error(response)
        else:
            logging.debug(">> Job started")

        return response

    def stop_job(self, job_id, params):
        stop_job_endpoint = "/".join([self.host, "api/v1/projects",
                                      self.username, self.project_name, "jobs", str(job_id), "stop"])
        res = requests.post(
            stop_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        logging.error(response)
        if (res.status_code != 201):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug(">> Job stopped")

        return response

    def create_model(self, params):
        create_model_endpoint = "/".join([self.host,
                                          "api/altus-ds-1", "models", "create-model"])
        res = requests.post(
            create_model_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug(">> Model created")

        return response
      
    def get_model(self, params):
        create_model_endpoint = "/".join([self.host,
                                          "api/altus-ds-1", "models", "get-model"])
        res = requests.post(
            create_model_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug(">> Got model")

        return response
      
    def list_models(self, params):
        create_model_endpoint = "/".join([self.host,
                                          "api/altus-ds-1", "models", "list-models"])
        res = requests.post(
            create_model_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug(">> Listing models")

        return response      

    def create_application(self, params):
        create_application_endpoint = "/".join([self.host, "api/v1/projects",
                                                self.username, self.project_name, "applications"])
        res = requests.post(
            create_application_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 201):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug(">> Application created")

        return response
      
    def get_all_applications(self, params):
        create_job_endpoint = "/".join([self.host, "api/v1/projects",
                                        self.username, self.project_name, "applications"])
        res = requests.get(
            create_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("List of applications retrieved")

        return response      
  
    def get_application(self, params, app_id):
        create_job_endpoint = "/".join([self.host, "api/v1/projects",
                                        self.username, self.project_name, "applications", app_id])
        res = requests.get(
            create_job_endpoint,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            data=json.dumps(params)
        )

        response = res.json()
        if (res.status_code != 200):
            logging.error(response["message"])
            logging.error(response)
        else:
            logging.debug("List of applications retrieved")

        return response
      
#    def run_expermiment(self, params):
#        create_model_endpoint = "/".join([self.host,
#                                          "api/altus-ds-1", "ds", "run"])
#        res = requests.post(
#            create_model_endpoint,
#            headers={"Content-Type": "application/json"},
#            auth=(self.api_key, ""),
#            data=json.dumps(params)
#        )
#
#        response = res.json()
#        if (res.status_code != 200):
#            logging.error(response["message"])
#            logging.error(response)
#        else:
#            logging.debug(">> Model created")
#
#        return response