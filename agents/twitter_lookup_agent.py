from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup(name):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """Given the name {name_of_person} i want you to find their twitter profile page, and 
        extract it from their username. In your final response only the persons username should show.
    """

    tools_for_agent = [
        Tool(
            name="Crawl google 4 twitter profile page",
            func=get_profile_url,
            description="useful for when you need to get the twitter page url",
        )
    ]
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    twitter_username = agent.run(prompt_template.format_prompt(name_of_person=name))
    return twitter_username
