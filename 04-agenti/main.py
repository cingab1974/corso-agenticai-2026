import os
import pickle
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai import LLM
from crewai.flow import Flow, start, listen
import chromadb
from docling_core.types.doc import DoclingDocument, SectionHeaderItem

# ------------------
# DATABASE
# ------------------

# Path to your ChromaDB directory
db_path = "/home/cla/OneDrive_unibo/WIP/corso_AI/laboratori/03-rag/rag_series/chroma_db_docling"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=db_path)

# Get your collection
collection = client.get_collection("rag_collection_docling_hf_chunker")


# ------------------
# DOCLING KNOWLEDGE BASE
# ------------------

docling_docs_path = "/home/cla/OneDrive_unibo/WIP/corso_AI/laboratori/03-rag/rag_series/processed_docs/"


# ------------------
# TOOLS
# ------------------


# Define and create a tool for the agent to query the database
@tool("Database Search")
def search_tool(query: str) -> str:
    """Use this tool to search for information in the ChromaDB database."""
    results = collection.query(query_texts=[query], n_results=5)
    # The query returns a list of lists of documents. We are interested in the first list.
    return "\n".join(results["documents"][0])


@tool("List Docling Documents")
def list_docling_documents(query: str = None) -> str:
    """
    Lists all available Docling documents in the knowledge base.
    This tool can be used to see which documents are available to read.
    """
    try:
        files = [f for f in os.listdir(docling_docs_path) if f.endswith(".pkl")]
        if not files:
            return "No Docling documents found."
        return "Available documents:\n- " + "\n- ".join(files)
    except FileNotFoundError:
        return f"Error: Directory not found at {docling_docs_path}"


@tool("Read Docling Document")
def read_docling_document(filename: str, start_element: int = 0, end_element: int = 20) -> str:
    """
    Reads content from a specific Docling document and returns it as markdown.
    Use 'list_docling_documents' to find available filenames.
    You can specify a range of elements to read; by default, it reads the first 20.
    """
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    file_path = os.path.join(docling_docs_path, filename)

    try:
        with open(file_path, "rb") as f:
            doc: DoclingDocument = pickle.load(f)

        # Export a slice of the document to markdown
        markdown_content = doc.export_to_markdown(
            from_element=start_element, to_element=end_element
        )
        return markdown_content
    except FileNotFoundError:
        return f"Error: Document '{filename}' not found."
    except Exception as e:
        return f"An error occurred while reading the document: {e}"


@tool("List Docling Document Sections")
def list_document_sections(filename: str) -> str:
    """
    Lists all the section titles from a specific Docling document.
    Use 'list_docling_documents' to find available filenames.
    """
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    file_path = os.path.join(docling_docs_path, filename)

    try:
        with open(file_path, "rb") as f:
            doc: DoclingDocument = pickle.load(f)

        sections = [
            item.text
            for item in doc.texts
            if isinstance(item, SectionHeaderItem)
        ]

        if not sections:
            return f"No sections found in document '{filename}'."

        return f"Sections in '{filename}':\n- " + "\n- ".join(sections)
    except FileNotFoundError:
        return f"Error: Document '{filename}' not found."
    except Exception as e:
        return f"An error occurred while reading the document sections: {e}"


@tool("Read Docling Document Section")
def read_document_section(filename: str, section_title: str) -> str:
    """
    Reads a specific section from a Docling document and returns its content as markdown.
    Use 'list_document_sections' to find the exact section titles.
    """
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    file_path = os.path.join(docling_docs_path, filename)

    try:
        with open(file_path, "rb") as f:
            doc: DoclingDocument = pickle.load(f)

        all_body_items = [item.resolve(doc) for item in doc.body.children]
        
        start_index = -1
        start_node_level = -1

        for i, item in enumerate(all_body_items):
            if isinstance(item, SectionHeaderItem) and item.text.strip().lower() == section_title.strip().lower():
                start_index = i
                start_node_level = item.level
                break
        
        if start_index == -1:
            return f"Error: Section '{section_title}' not found in document '{filename}'."

        end_index = len(all_body_items)
        for i in range(start_index + 1, len(all_body_items)):
            item = all_body_items[i]
            if isinstance(item, SectionHeaderItem) and item.level <= start_node_level:
                end_index = i
                break
        
        section_items = all_body_items[start_index:end_index]

        temp_doc = DoclingDocument(name=f"section_{section_title}")
        temp_doc.add_node_items(node_items=section_items, doc=doc)

        return temp_doc.export_to_markdown()

    except FileNotFoundError:
        return f"Error: Document '{filename}' not found."
    except Exception as e:
        return f"An error occurred while reading the document section: {e}"

# ------------------
# LLM
# ------------------
llm = LLM(
    model="openai/vertex_ai/mistral-small-2503",
    base_url="https://litellm-proxy-1013932759942.europe-west8.run.app",
    api_key="",
)

# ------------------
# AGENTS
# ------------------

# Researcher Agent
researcher = Agent(
    llm=llm,
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change "
        "the world."
    ),
    tools=[search_tool],
    allow_delegation=True,
)

# Writer Agent
writer = Agent(
    llm=llm,
    role="Writer",
    goal="Narrate compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to a wider audience."
    ),
    tools=[],
    allow_delegation=False,
)

# Coordinator Agent
coordinator = Agent(
    llm=llm,
    role="Coordinator",
    goal="Coordinate the research and writing process to produce a comprehensive report on {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "As the orchestrator of the team, you ensure a seamless"
        "workflow between the researcher and writer, guaranteeing the"
        "final report is cohesive, accurate, and delivered on time."
    ),
    tools=[],
    allow_delegation=True,
)

# ------------------
# TASKS
# ------------------

# Research Task
research_task = Task(
    description=(
        "Identify the latest trends and breakthroughs in {topic}. "
        "Focus on primary sources and groundbreaking studies. "
        "Your final report should be a detailed summary of your findings."
    ),
    expected_output="A comprehensive rapport with links to the sources of information on the topic {topic}.",
    tools=[search_tool],
    agent=researcher,
)

# Writing Task
writing_task = Task(
    description=(
        "Using the research findings, write a compelling and easy-to-understand"
        "article on {topic}. Your writing should be engaging and"
        "accessible to a broad audience. Avoid jargon and focus on the real-world"
        "implications of the technologies."
    ),
    expected_output="A 4 paragraphs article about the topic {topic}.",
    tools=[],
    agent=writer,
    output_file="report.md",
)

# Coordination Task
coordination_task = Task(
    description=(
        "Oversee the research and writing tasks. Ensure the researcher provides"
        "high-quality, relevant information, and the writer crafts a narrative"
        "that is both informative and engaging. Review the final article for"
        "accuracy and clarity."
    ),
    expected_output="A final, polished article on {topic}, ready for publication.",
    agent=coordinator,
)

# ------------------
# FLOW
# ------------------


class ReportGenerationFlow(Flow):
    @start()
    def research(self):
        """
        Starts the research process for the given topic.
        The topic is expected to be in self.state['topic'].
        """
        topic = self.state["topic"]
        # Create a fresh agent for this task to use the docling tools
        researcher_agent = Agent(
            llm=llm,
            role="Senior Researcher",
            goal=f"Uncover groundbreaking technologies in {topic}",
            verbose=True,
            memory=True,
            backstory=(
                "Driven by curiosity, you're at the forefront of"
                "innovation, eager to explore and share knowledge that could change "
                "the world."
            ),
            tools=[
                list_docling_documents,
                read_docling_document,
                list_document_sections,
                read_document_section,
            ],
            allow_delegation=False,
        )
        task = Task(
            description=f"Identify the latest trends and breakthroughs in {topic}. "
            "Focus on primary sources and groundbreaking studies. "
            "Your final report should be a detailed summary of your findings, "
            "and include links to the sources of information.",
            expected_output=f"A detailed summary of findings on {topic}.",
            agent=researcher_agent,
        )
        # Execute the task directly with the agent
        research_result = researcher_agent.execute_task(task)
        return research_result

    @listen(research)
    def write(self, research_summary):
        """
        Takes the research summary, writes an article, and returns the final report.
        """
        topic = self.state["topic"]
        # Update the goal of the global agent to avoid state pollution
        writer.goal = f"Narrate compelling tech stories about {topic}"
        task = Task(
            description=f"Using the following research findings:\n---\n{research_summary}\n---\n\n"
            f"Write a compelling and easy-to-understand 4-paragraph article on {topic}. "
            "Your writing should be engaging and accessible to a broad audience. "
            "Avoid jargon and focus on the real-world implications of the technologies.",
            expected_output=f"A 4-paragraph article about {topic}.",
            agent=writer,
        )
        # Execute the task directly with the agent
        final_report = writer.execute_task(task)

        # Manually save the report to a file
        if final_report:
            with open("report.md", "w") as f:
                f.write(final_report)

        return final_report


# ------------------
# CREW
# ------------------

# Create the crew for the sequential process
crew = Crew(
    agents=[researcher, writer, coordinator],
    tasks=[research_task, writing_task, coordination_task],
    process=Process.sequential,
)

# ------------------
# KICKOFF
# ------------------

# Start the selected workflow
if __name__ == "__main__":
    topic = input("Enter the topic for the report: ")
    workflow_choice = input("Choose workflow: 'crew' or 'flow': ").lower()

    if workflow_choice == "crew":
        print("Running Crew workflow...")
        result = crew.kickoff(inputs={"topic": topic})
    elif workflow_choice == "flow":
        print("Running Flow workflow...")
        flow = ReportGenerationFlow()
        # Set initial state before kicking off the flow
        flow.state["topic"] = topic
        result = flow.kickoff()
    else:
        result = "Invalid choice. Please enter 'crew' or 'flow'."

    print("\n######################\n")
    print(result)
