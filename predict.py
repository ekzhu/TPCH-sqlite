import argparse
import json
import os
import re
import subprocess
import random
from typing import List, Union
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

_sqlite3_runtime_reg = re.compile(
    r"Run Time: real (\d+\.\d+) user (\d+\.\d+) sys (\d+\.\d+)"
)


def _execute_queries(db_path: str, query_files: List[str], timeout: int = 60):
    """Execute SQL queries on the SQLite database and observe the runtimes."""
    run_times = []
    for query_file in query_files:
        # Execute the query using the sqlite3 command line tool and capture the output using subprocess.
        try:
            result = subprocess.run(
                ["sqlite3", db_path],
                stdin=open(query_file),
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Error executing query {query_file}: {result.stderr}")
                continue
        except subprocess.TimeoutExpired:
            print(f"Query {query_file} timed out after {timeout} seconds")
            continue
        # Extract the run times from the output.
        match = _sqlite3_runtime_reg.search(result.stdout)
        if match:
            real, user, sys = match.groups()
            run_times.append(
                {
                    "real": float(real),
                    "user": float(user),
                    "sys": float(sys),
                    "query": open(query_file).read(),
                }
            )
            print(f"Runtimes for {query_file}: {run_times[-1]['real']}")
        else:
            raise ValueError(
                f"Could not extract run times from output: {result.stdout}"
            )
    return run_times


def _get_schema(db_path: str) -> str:
    """Get the schema of the database."""
    output = os.popen(f"sqlite3 {db_path} .schema").read()
    return output


_row_count_reg = re.compile(r"(\d+)")


def _get_row_counts(db_path: str) -> dict:
    """Get the row counts of the tables in the database."""
    output = os.popen(f"sqlite3 {db_path} .tables").read()
    tables = output.split()
    row_counts = {}
    for table in tables:
        output = os.popen(f"sqlite3 {db_path} 'SELECT COUNT(*) FROM {table}'").read()
        match = _row_count_reg.search(output)
        if match:
            row_counts[table] = int(match.group(1))
        else:
            raise ValueError(f"Could not extract row count from output: {output}")
    return row_counts


def _system_prompt(schema: str, row_counts: dict) -> str:
    """Create a system prompt for the assistant."""
    row_count_str = "\n".join([f"{table}: {row_counts[table]}" for table in row_counts])
    template = f"""You are responsible for estimating the running time of SQL queries in a SQLite database. You are given a SQL query and output an estimate running time in seconds. The database has the following schema:
{schema}
The row counts of the tables are:
{row_count_str}
""".format(
        schema=schema, row_counts=row_count_str
    )
    return SystemMessagePromptTemplate.from_template(template)


def _examples(run_times: list):
    """Create examples of the assistant's responses."""
    examples = []
    for run_time in run_times:
        query = run_time["query"]
        real = run_time["real"]
        example_human = SystemMessagePromptTemplate.from_template(
            query, additional_kwargs={"name": "example_user"}
        )
        example_ai = SystemMessagePromptTemplate.from_template(
            f"{real}", additional_kwargs={"name": "example_assistant"}
        )
        examples += [example_human, example_ai]
    return examples


def _chat_prompt(
    schema: str,
    row_counts: dict,
    run_times: list,
) -> ChatPromptTemplate:
    """Create a chat prompt for the assistant."""
    system_prompt = _system_prompt(schema, row_counts)
    examples = _examples(run_times)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt] + examples + [human_message_prompt]
    )
    return chat_prompt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--db-path", type=str, required=True, help="Path to the SQLite database."
)
parser.add_argument(
    "--query-files", type=str, nargs="+", required=True, help="Path to the query files."
)
parser.add_argument(
    "--runtime-cache", type=str, default=None, help="Path to cache the run times."
)
parser.add_argument(
    "--timeout",
    type=int,
    default=60,
    help="Timeout in seconds for observing run times.",
)
parser.add_argument(
    "--num-examples",
    type=int,
    default=2,
    help="Number of examples to use for in-context learning.",
)
parser.add_argument("--debug", action="store_true", help="Debug mode.")
parser.add_argument(
    "--output", type=str, required=True, help="Path to the test results."
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use.")
args = parser.parse_args()

if args.runtime_cache and os.path.exists(args.runtime_cache):
    run_times = json.load(open(args.runtime_cache))
else:
    run_times = _execute_queries(args.db_path, args.query_files, timeout=args.timeout)
    if args.runtime_cache:
        with open(args.runtime_cache, "w") as f:
            json.dump(run_times, f, indent=2, sort_keys=True)

# Split the run times in to examples and test cases.
random.seed(args.seed)
random.shuffle(run_times)
example_run_times = run_times[: args.num_examples]
test_run_times = run_times[args.num_examples :]

schema = _get_schema(args.db_path)
row_counts = _get_row_counts(args.db_path)

chat = ChatOpenAI(temperature=0, model_name=args.model)

chat_prompt = _chat_prompt(schema, row_counts, example_run_times)

if args.debug:
    for message in chat_prompt.messages:
        print(message)

chain = LLMChain(llm=chat, prompt=chat_prompt)

results = []
for test_run_time in test_run_times:
    query = test_run_time["query"]
    real = test_run_time["real"]
    estimate = chain.run(query)
    results.append({"query": query, "real": real, "estimate": estimate})
with open(args.output, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)
