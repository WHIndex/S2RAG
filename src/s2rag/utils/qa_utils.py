from typing import Dict, List, Union, Any, Optional


from ..prompts.prompt_template_manager import PromptTemplateManager
from .logging_utils import get_logger
from ..llm.compatible_chat import CachedChatClient

logger = get_logger(__name__)



def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}


    for element in elements:

        lines = element.split('\n')
        first_line = lines[0]


        if first_line in merged_dict:

            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:

            merged_dict[first_line] = prefix + element


    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, prompt_template_manager: PromptTemplateManager, query: str, passages: list, thoughts: list, llm_client: CachedChatClient):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with remote language models. The generated thought is used for further retrieval step.
    :return: next thought
    """

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)


    messages = prompt_template_manager.render(name=f'ircot_{dataset}', prompt_user=prompt_user)

    try:
        response_message, metadata = llm_client.infer(messages=messages)
        response_content = response_message[0]["content"]
    except Exception as e:
        logger.exception("An exception occurred while calling LLM for QA!")
        return ''

    return response_content
