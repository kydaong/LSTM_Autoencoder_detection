# Created by aaronkueh at 5/4/24
import json
import requests
import yaml
import logging
from aom.definitions import CONFIG_DIR


class ApiClient:

    def __init__(self, config_file: str = '', debug: bool = False):
        logging.basicConfig()
        self._logger = logging.getLogger(__class__.__name__)
        self.url = None
        self.api_key = None
        self.configure(config_file)
        self.debug = debug
        self.proxies = dict()

    def configure(self, config_file):
        """
        Configures the client using a config file
        Args:
            config_file: the path of the config file, relative or absolute
        """

        with open(config_file) as stream:
            try:
                configs = yaml.safe_load(stream)

                if configs.get('debug') is True:
                    self._logger.setLevel(logging.DEBUG)

                if 'proxy' in configs:
                    self.proxies = {'https': configs['proxy']}

                # Copy the authentication information
                self.api_key = configs.get('api_key')
                self.url = configs.get('host_url')
            except yaml.YAMLError as e:
                print(e)

    def get_request(self, api, parameters: dict = None) -> json:
        """
        Worker function to perform the GET request
        Args:
            api: API key
            parameters:  API body in dictionary format
        Returns:
            json: GET response
        """

        # Set up headers
        headers = dict()
        headers['appKey'] = self.api_key
        headers['Accept'] = 'application/json'
        # url = self.url + api
        url = self.url


        # Send the request and get the response
        response = requests.get(url, headers=headers, params=parameters, proxies=self.proxies, verify=False)

        # Parse the response into json format
        response_json = json.loads(response.content)

        # Return None if the response code indicates an error
        if response.status_code != 200:
            print('Error: Response Code', str(response.status_code), response_json)
            return None

        return response_json

    def post_request(self, api, parameters: dict = None) -> json:
        """
        Worker function to perform the POST request
        Args:
            api: API key
            parameters:  API body in dictionary format
        Returns:
            json: POST response
        """

        # Set up headers & bodies
        headers = dict()
        headers['appKey'] = self.api_key
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'

        # Set up the URL
        url = self.url + api

        # Send the request and get the response
        response = requests.post(url, headers=headers, params=parameters)
        # print('Response:', response.text)

        # Parse the response into json format
        response_json = json.loads(response.content)

        # Return None if the response code indicates an error
        if response.status_code != 200:
            print('Error: Response Code', str(response.status_code), response_json)
            return None

        return response_json

    def put_request(self, api, parameters: json = None) -> json:
        """
        Worker function to perform the PUT request
        Args:
            api: API key
            parameters:  API body in dictionary format
        Returns:
            json: PUT response
        """

        # Set up headers & bodies
        headers = dict()
        headers['appKey'] = self.api_key
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'

        # Set up the URL
        url = self.url + api

        # Send the request and get the response
        response = requests.put(url, headers=headers, data=parameters)
        # print('Response:', response.text)

        # Parse the response into json format
        response_json = json.loads(response.content)

        # Return None if the response code indicates an error
        if response.status_code != 200:
            print('Error: Response Code', str(response.status_code), response_json)
            return None

        return response_json


if __name__ == '__main__':
    config = f'{CONFIG_DIR}/api_config.yaml'
    client = ApiClient(config_file=config)
    api_input = dict()
    post_req = client.get_request(api=None, parameters=None)
    print(post_req)