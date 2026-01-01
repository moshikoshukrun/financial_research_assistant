"""Main entry point for the financial research assistant CLI."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.agent import FinancialAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Rich console for pretty output
console = Console()


def display_response(response: dict):
    """
    Display agent response with formatting.
    
    Args:
        response: Response dictionary from agent
    """
    # Display answer
    console.print("\n")
    console.print(Panel(
        response['answer'],
        title="Answer",
        border_style="green"
    ))
    
    # Display citations
    if response['citations']:
        console.print("\n[bold cyan]Citations:[/bold cyan]")
        for i, citation in enumerate(response['citations'], 1):
            if citation['source_type'] == 'document':
                console.print(
                    f"[{i}] Section: {citation['section']}, "
                    f"Page: {citation['page']}"
                )
                console.print(f"    {citation['text'][:150]}...\n")
            else:
                console.print(f"[{i}] {citation['title']}")
                console.print(f"    URL: {citation['url']}")
                console.print(f"    {citation['text'][:150]}...\n")
    
    # Display tools used
    if response.get('tools_used'):
        console.print(f"\n[dim]Tools used: {', '.join(response['tools_used'])}[/dim]")
    
    # Display errors if any
    if response.get('errors'):
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in response['errors']:
            console.print(f"  - {error}")


def main():
    """Main CLI function."""
    # Load environment variables
    load_dotenv()
    
    # Display welcome message
    console.print(Panel(
        "[bold blue]Financial Research Assistant[/bold blue]\n"
        "Ask questions about Apple Inc.'s 10-K filing",
        border_style="blue"
    ))
    
    # Check for document
    document_path = "data/apple_10k_2023.htm"
    if not Path(document_path).exists():
        console.print("[bold red]Error:[/bold red] 10-K document not found!")
        console.print(f"Expected location: {document_path}")
        console.print("\nPlease download Apple's 10-K from SEC EDGAR and place it in the data/ folder.")
        console.print("URL: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193")
        sys.exit(1)
    
    # Check for API keys
    required_keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY"]
    
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        console.print("[bold red]Error:[/bold red] Missing required API keys:")
        for key in missing_keys:
            console.print(f"  - {key}")
        console.print("\nPlease set these in your .env file (see .env.example)")
        sys.exit(1)
    
    # Initialize agent
    try:
        console.print("\n[cyan]Initializing agent...[/cyan]")
        agent = FinancialAgent(document_path)
        console.print("[green]Agent initialized successfully![/green]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize agent:[/bold red] {str(e)}")
        sys.exit(1)
    
    # Main interaction loop
    console.print("[dim]Type 'exit' or 'quit' to end the session[/dim]\n")
    
    while True:
        try:
            # Get user input
            query = console.input("[bold green]>[/bold green] ")
            
            if not query.strip():
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("\n[cyan]Thank you for using Financial Research Assistant![/cyan]")
                break
            
            # Process query
            console.print("\n[cyan]Processing...[/cyan]")
            response = agent.answer_query(query)
            
            # Display response
            display_response(response)
            console.print("\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[cyan]Thank you for using Financial Research Assistant![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
            logger.exception("Error in main loop")


if __name__ == "__main__":
    main()
