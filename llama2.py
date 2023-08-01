from abc import abstractmethod
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
from typing import Any, Dict, Generic, List, Mapping, Optional, TypeVar, Union
from pydantic import Extra, root_validator
from langchain.chains import APIChain
from langchain import LLMChain
from langchain.requests import TextRequestsWrapper
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from asyncer import asyncify
from langchain.callbacks.manager import CallbackManagerForLLMRun

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
import taapi_docs
import cmc_api_docs
import os
import asyncio 

INPUT_TYPE = TypeVar("INPUT_TYPE", bound=Union[str, List[str]])
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", bound=Union[str, List[List[float]]])


class ContentHandlerBase(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    """A handler class to transform input from LLM to a
    format that SageMaker endpoint expects. Similarily,
    the class also handles transforming output from the
    SageMaker endpoint to a format that LLM class expects.
    """

    """
    Example:
        .. code-block:: python

            class ContentHandler(ContentHandlerBase):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')
                
                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    content_type: Optional[str] = "text/plain"
    """The MIME type of the input data passed to endpoint"""

    accepts: Optional[str] = "text/plain"
    """The MIME type of the response data returned from endpoint"""

    @abstractmethod
    def transform_input(self, prompt: INPUT_TYPE, model_kwargs: Dict) -> bytes:
        """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """

    @abstractmethod
    def transform_output(self, output: bytes) -> OUTPUT_TYPE:
        """Transforms the output from the model to string that
        the LLM class expects.
        """


class LLMContentHandler(ContentHandlerBase[str, str]):
    """Content handler for LLM class."""

class SagemakerLLM(LLM):
    """Wrapper around custom Sagemaker Inference Endpoints.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Sagemaker endpoint.
    See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html
    """

    """
    Example:
        .. code-block:: python

            from langchain import SagemakerEndpoint
            endpoint_name = (
                "my-endpoint-name"
            )
            region_name = (
                "us-west-2"
            )
            credentials_profile_name = (
                "default"
            )
            se = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )
    """
    client: Any  #: :meta private:

    endpoint_name: str = ""
    """The name of the endpoint from the deployed Sagemaker model.
    Must be unique within an AWS Region."""

    region_name: str = ""
    """The aws region where the Sagemaker model is deployed, eg. `us-west-2`."""

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    content_handler: LLMContentHandler
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    """
     Example:
        .. code-block:: python

        from langchain.llms.sagemaker_endpoint import LLMContentHandler

        class ContentHandler(LLMContentHandler):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')
                
                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
    function. See `boto3`_. docs for more info.
    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""
        try:
            import boto3

            try:
                if values["credentials_profile_name"] is not None:
                    session = boto3.Session(
                        profile_name=values["credentials_profile_name"]
                    )
                else:
                    # use default credentials
                    session = boto3.Session()

                values["client"] = session.client(
                    "sagemaker-runtime", region_name=values["region_name"]
                )

            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        except ImportError:
            raise ValueError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.endpoint_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sagemaker_endpoint"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call out to Sagemaker inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        if run_manager:
            run_manager.on_text(prompt, color="yellow", end="\n", verbose=self.verbose)
        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        # send request
        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accepts,
                **_endpoint_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        text = self.content_handler.transform_output(response["Body"])
        # sta=text.find(prompt)+len(prompt)
        # text=text[sta:]
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to the sagemaker endpoint.
            text = enforce_stop_tokens(text, stop)
        if run_manager:
            run_manager.on_text(text=text, color="yellow", end="\n", verbose=self.verbose)
        return text 

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call out to Sagemaker inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        if run_manager:
            await run_manager.on_text(prompt, color="yellow", end="\n", verbose=self.verbose)
        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        # send request
        try:
            response = await asyncify(self.client.invoke_endpoint)(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accepts,
                **_endpoint_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        text = self.content_handler.transform_output(response["Body"])
        # sta=text.find(prompt)+len(prompt)
        # text=text[sta:]
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to the sagemaker endpoint.
            text = enforce_stop_tokens(text, stop)
        if run_manager:
            await run_manager.on_text(text=text, color="yellow", end="\n", verbose=self.verbose)
        return text 

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


class IndicatorsQuestionsChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
            run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text(
                response.generations[0][0].text,
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
            await run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text(
                response.generations[0][0].text,
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "indicators_questions_chain"

    @classmethod
    def from_indicators(
        cls,
        indicators: str,
        **kwargs: Any,
    ) -> Chain:
        PROMPT_TEMPLATE = (
            """The user's question may be to ask the latest market trend of a certain cryptocurrency. 
The following index tools can help users analyze market trend.
Index tool's names are """
            + indicators
        )
        PROMPT_TEMPLATE = (
            PROMPT_TEMPLATE
            + """\nPlease generats questions, to ask the latest index above of the cryptocurrency with its symbol in user's question.
Note that some index are composed of a set of numerical values. So you should generate the questions ask to get all values of the index.
You should generate the questions in JSON Object format, and use index tool's name as JSON object's key, use the question which you generated as the value of JSON object.
Do not need any hint, just a JSON object.
User's Question: {question}
You generations:"""
        )
        # Finally, add a item into the JSON object, key is "question", use the following user's question as its value.

        prompt = PromptTemplate(input_variables=["question"], template=PROMPT_TEMPLATE)
        content_handler = ContentHandler()
        llm = SagemakerLLM(
            endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-07-23-15-934",
            # credentials_profile_name="default",
            region_name="us-east-1",
            model_kwargs={
                "parameters": {
                    "do_sample": True,
                    # "top_p": 0.9,
                    # "top_k": 10,
                    "repetition_penalty": 1.03,
                    "max_new_tokens": 1024,
                    "temperature": 0.1,
                    "return_full_text": False,
                    # "max_length":2048,
                    "truncate": 2048,
                    # "num_return_sequences":2000,
                    # "stop": ["\n"],
                },
            },
            content_handler=content_handler,
            verbose=kwargs["verbose"],
        )
        return cls(llm=llm, prompt=prompt, **kwargs)


class Llama2APIChain(APIChain):
    @property
    def _chain_type(self) -> str:
        return "llama2_api_chain"

    @classmethod
    def from_docs(
        cls,
        api_docs: str,
        headers: any,
        **kwargs: any,
    ) -> APIChain:
        content_handler = ContentHandler()

        llm_gen_url = SagemakerLLM(
            endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-07-23-15-934",
            # credentials_profile_name="default",
            region_name="us-east-1",
            model_kwargs={
                "parameters": {
                    "do_sample": True,
                    # "top_p": 0.9,
                    # "top_k": 10,
                    "repetition_penalty": 1.03,
                    "max_new_tokens": 1024,
                    "temperature": 0.1,
                    "return_full_text": False,
                    # "max_length":2048,
                    "truncate": 2048,
                    # "num_return_sequences":2000,
                    # "stop": ["\n"],
                },
            },
            content_handler=content_handler,
            # verbose=kwargs["verbose"],
        )

        llm_gen_resp = SagemakerLLM(
            endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-07-23-15-934",
            # credentials_profile_name="default",
            region_name="us-east-1",
            model_kwargs={
                "parameters": {
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 10,
                    "repetition_penalty": 1.03,
                    "max_new_tokens": 1024,
                    "temperature": 0.9,
                    "return_full_text": False,
                    # "max_length":2048,
                    "truncate": 2048,
                    # "num_return_sequences":2000,
                    # "stop": ["\nHuman:"],
                },
            },
            content_handler=content_handler,
            # verbose=kwargs["verbose"],
        )

        API_URL_PROMPT_TEMPLATE = """API Documentation:
{api_docs}

According above documentation, the full API url was generated to call for answering the question bellow.
The full API url in order to get a response that is as short as possible. And the full API url to deliberately exclude any unnecessary pieces of data in the API call.
You should not generate any hint. Just generate the full API url to answer the question as far as possible.

Question:{question}

API url:"""
        API_URL_PROMPT = PromptTemplate(
            input_variables=[
                "api_docs",
                "question",
            ],
            template=API_URL_PROMPT_TEMPLATE,
        )
        API_RESPONSE_PROMPT_TEMPLATE = (
            API_URL_PROMPT_TEMPLATE
            + """ {api_url}
    
Here is the response from the API:

{api_response}

Summarize this response to answer the original question.
Please include all data in the response.

Summary:"""
        )
        API_RESPONSE_PROMPT = PromptTemplate.from_template(API_RESPONSE_PROMPT_TEMPLATE)
        return cls(
            api_request_chain=LLMChain(
                llm=llm_gen_url, prompt=API_URL_PROMPT, **kwargs
            ),
            api_answer_chain=LLMChain(
                llm=llm_gen_resp, prompt=API_RESPONSE_PROMPT, **kwargs
            ),
            api_docs=api_docs,
            requests_wrapper=TextRequestsWrapper(headers=headers),
            **kwargs,
        )


class MarketTrendChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
            run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text(
                response.generations[0][0].text,
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
            await run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text(
                response.generations[0][0].text,
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "indicators_questions_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> Chain:
        PROMPT_TEMPLATE = """Content
```
{data}
```

Question
```
{question}
```

Please organize the above content into a complete article. The required content includes answers to questions using the above content, as well as analysis of the above market data and investment suggestions.
Here is the article:
"""
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        return cls(llm=llm, prompt=prompt, **kwargs)

class MarketTrendAndInvestmentAdviseToolChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    iq_chain:IndicatorsQuestionsChain

    rsi_chain : Llama2APIChain
    cci_chain : Llama2APIChain
    dmi_chain : Llama2APIChain
    psar_chain : Llama2APIChain
    stochrsi_chain : Llama2APIChain
    cmf_chain : Llama2APIChain

    latest_quote_chain:Llama2APIChain

    mt_chain:MarketTrendChain

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
             run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response =  self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
             run_manager.on_text(
                response.generations[0][0].text,
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
        question=response.generations[0][0].text
        indicator_questions_json =  self.iq_chain.run(question)
        index_questions = json.loads(indicator_questions_json)
        
        rsi_res =  self.rsi_chain.run(index_questions["RSI"])
        cci_res =  self.cci_chain.run(index_questions["CCI"])
        dmi_res =  self.dmi_chain.run(index_questions["DMI"])
        psar_res =  self.psar_chain.run(index_questions["PSAR"])
        stochrsi_res =  self.stochrsi_chain.run(index_questions["STOCHRSI"])
        cmf_res =  self.cmf_chain.run(index_questions["CMF"])
        latest_quote_res= self.latest_quote_chain.run(question)
        
        data=rsi_res+"\n"+cci_res+"\n"+dmi_res+"\n"+psar_res+"\n"+stochrsi_res+"\n"+cmf_res+"\n"+latest_quote_res
        market_trend_res= self.mt_chain.run(question=question,data=data)

        return {self.output_key: market_trend_res}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        if run_manager:
            await run_manager.on_text(
                prompt_value.to_string(), color="green", end="\n", verbose=self.verbose
            )
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text(
                response.generations[0][0].text,
                color="yellow",
                end="\n",
                verbose=self.verbose,
            )
        question=response.generations[0][0].text
        indicator_questions_json = await self.iq_chain.arun(question)
        index_questions = json.loads(indicator_questions_json)

        tasks=[
         self.rsi_chain.arun(index_questions["RSI"]),
         self.cci_chain.arun(index_questions["CCI"]),
         self.dmi_chain.arun(index_questions["DMI"]),
         self.psar_chain.arun(index_questions["PSAR"]),
         self.stochrsi_chain.arun(index_questions["STOCHRSI"]),
         self.cmf_chain.arun(index_questions["CMF"]),
         self.latest_quote_chain.arun(question),
        ] 
        resp=await asyncio.gather(*tasks)

        data="\n".join(resp)
        market_trend_res=await self.mt_chain.arun(question=question,data=data)
        return {self.output_key: market_trend_res}

    @property
    def _chain_type(self) -> str:
        return "market_trend_investment_advise_chain"

    @classmethod
    def from_create(
        cls,
        **kwargs: Any,
    ) -> Chain:
        prompt_template = (
"""User may ask some cryptocurrency's market trend.
Please generate a complete question in English, using user's input below.
User's input:{user_input}
Complete question:"""
)
        content_handler = ContentHandler()
        llama2_01 = SagemakerLLM(
            endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-07-23-15-934",
            # credentials_profile_name="default",
            region_name="us-east-1",
            model_kwargs={
                "parameters": {
                    "do_sample": True,
                    # "top_p": 0.9,
                    # "top_k": 10,
                    "repetition_penalty": 1.03,
                    "max_new_tokens": 1024,
                    "temperature": 0.1,
                    "return_full_text": False,
                    # "max_length":2048,
                    "truncate": 2048,
                    # "num_return_sequences":2000,
                    # "stop": ["\n"],
                },
            },
            content_handler=content_handler,
            verbose=kwargs["verbose"],
        )
        llama2_09 = SagemakerLLM(
            endpoint_name="huggingface-pytorch-tgi-inference-2023-07-24-07-23-15-934",
            # credentials_profile_name="default",
            region_name="us-east-1",
            model_kwargs={
                "parameters": {
                    "do_sample": True,
                    # "top_p": 0.9,
                    # "top_k": 10,
                    "repetition_penalty": 1.03,
                    "max_new_tokens": 2048,
                    "temperature": 0.9,
                    "return_full_text": False,
                    # "max_length":2048,
                    "truncate": 2048,
                    # "num_return_sequences":2000,
                    # "stop": ["\n"],
                },
            },
            content_handler=content_handler,
            verbose=kwargs["verbose"],
        )
        prompt = PromptTemplate.from_template(prompt_template)

        iq_chain = IndicatorsQuestionsChain.from_indicators(
            indicators="RSI,CCI,DMI,PSAR,STOCHRSI,CMF", **kwargs, 
        )

        taapi_key = os.getenv("TAAPI_KEY")
        headers = {
            "Accepts": "application/json",
        }

        rsi_api_docs = PromptTemplate.from_template(taapi_docs.RSI_API_DOCS).format(
            taapi_key=taapi_key
        )
        cci_api_docs = PromptTemplate.from_template(taapi_docs.CCI_API_DOCS).format(
            taapi_key=taapi_key
        )
        dmi_api_docs = PromptTemplate.from_template(taapi_docs.DMI_API_DOCS).format(
            taapi_key=taapi_key
        )
        psar_api_docs = PromptTemplate.from_template(taapi_docs.PSAR_API_DOCS).format(
            taapi_key=taapi_key
        )
        stochrsi_api_docs = PromptTemplate.from_template(taapi_docs.STOCHRSI_API_DOCS).format(
            taapi_key=taapi_key
        )
        cmf_api_docs = PromptTemplate.from_template(taapi_docs.CMF_API_DOCS).format(
            taapi_key=taapi_key
        )
        rsi_chain = Llama2APIChain.from_docs(
            api_docs=rsi_api_docs, headers=headers,**kwargs, 
        )
        cci_chain = Llama2APIChain.from_docs(
            api_docs=cci_api_docs, headers=headers, **kwargs,
        )
        dmi_chain = Llama2APIChain.from_docs(
            api_docs=dmi_api_docs, headers=headers, **kwargs,
        )
        psar_chain = Llama2APIChain.from_docs(
            api_docs=psar_api_docs, headers=headers, **kwargs,
        )
        stochrsi_chain = Llama2APIChain.from_docs(
            api_docs=stochrsi_api_docs, headers=headers, **kwargs,
        )
        cmf_chain = Llama2APIChain.from_docs(
            api_docs=cmf_api_docs, headers=headers, **kwargs,
        )

        headers = {
          'Accepts': 'application/json',
          'X-CMC_PRO_API_KEY': os.getenv("CMC_API_KEY"),
        }
        latest_quote_chain = Llama2APIChain.from_docs(
            api_docs=cmc_api_docs.CMC_QUOTE_LASTEST_API_DOC, headers=headers, **kwargs,
        )

        gpt4 = ChatOpenAI(
            model="gpt-4",
            temperature=0.9,
            **kwargs,
        )
        mt_chain=MarketTrendChain.from_llm(llm=llama2_09,**kwargs)

        return cls(
            llm=llama2_01,
            iq_chain=iq_chain, 
            rsi_chain=rsi_chain,
            cci_chain=cci_chain,
            dmi_chain=dmi_chain,
            psar_chain=psar_chain,
            stochrsi_chain=stochrsi_chain,
            cmf_chain=cmf_chain,
            latest_quote_chain=latest_quote_chain,
            mt_chain=mt_chain,
            prompt=prompt, 
            **kwargs,)