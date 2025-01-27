from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (               
  solver, chain, prompt_template, generate, self_critique
)                                          

DEFAULT_PROMPT="{prompt}"

multihop_dataset = json_dataset(
    "dataset/morehopqa_final_150samples.json",
    FieldSpec(
        input="question",
        target="answer",
        id="_id",
        metadata=["answer_type", "no_of_hops", "reasoning_type"],
    ),
)

@solver 
def basic():
    return chain(
        prompt_template(DEFAULT_PROMPT), 
        generate(), 
    )

@solver 
def critique():
    return chain(
        prompt_template(DEFAULT_PROMPT), 
        generate(), 
        self_critique()
    )

@solver
def chain_of_thought():
    return chain(
        chain_of_thought(),
        generate(), 
        self_critique()
    )

@task
def multihop():
    return Task(  
        dataset=multihop_dataset,
        solver=basic(),
        scorer=model_graded_fact()
    )