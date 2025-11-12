"""
Streamlit visualization for TheoremForge database.

This script displays TheoremForgeState documents as the root and their associated traces.
"""

import asyncio
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from theoremforge.db import MongoDBClient


# Configure Streamlit page
st.set_page_config(
    page_title="TheoremForge Trace Visualization",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .state-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .trace-section {
        padding: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin-left: 1rem;
        margin-bottom: 1rem;
    }
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .error-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


async def get_connected_client():
    """Create a fresh MongoDB client and connect it in the current event loop."""
    # Create a new client for each async operation to avoid event loop conflicts
    client = MongoDBClient()

    try:
        await client.connect()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

    return client


async def fetch_states() -> List[Dict[str, Any]]:
    """Fetch all TheoremForgeState documents."""
    client = await get_connected_client()

    states = await client.theoremforgestate.find_many({})
    # Sort by creation time, most recent first
    states.sort(key=lambda x: x.get("createdAt", datetime.min), reverse=True)
    return states


async def fetch_root_states() -> List[Dict[str, Any]]:
    """Fetch only root TheoremForgeState documents (no parent)."""
    client = await get_connected_client()

    # Query for states where parentId is None or doesn't exist
    states = await client.theoremforgestate.find_many({})
    # Filter for root states (no parent_id)
    root_states = [s for s in states if not s.get("parentId")]
    # Sort by creation time, most recent first
    root_states.sort(key=lambda x: x.get("createdAt", datetime.min), reverse=True)
    return root_states


async def fetch_traces_for_state(state_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all traces associated with a state."""
    client = await get_connected_client()

    # Note: Using 'stateId' to match the database field name
    traces = {
        "prover": await client.provertrace.find_many({"stateId": state_id}),
        "self_correction": await client.selfcorrectiontrace.find_many(
            {"stateId": state_id}
        ),
        "proof_correction": await client.proofcorrectiontrace.find_many(
            {"stateId": state_id}
        ),
        "sketch_correction": await client.sketchcorrectiontrace.find_many(
            {"stateId": state_id}
        ),
        "correctness_check": await client.correctnesschecktrace.find_many(
            {"stateId": state_id}
        ),
        "shallow_solve": await client.shallowsolvetrace.find_many(
            {"stateId": state_id}
        ),
        "theorem_retrieval": await client.theoremretrievaltrace.find_many(
            {"stateId": state_id}
        ),
        "definition_retrieval": await client.definitionretrievaltrace.find_many(
            {"stateId": state_id}
        ),
        "informal_proof": await client.informalprooftrace.find_many(
            {"stateId": state_id}
        ),
        "proof_sketch": await client.proofsketchtrace.find_many({"stateId": state_id}),
        "proof_assembly": await client.proofassemblytrace.find_many(
            {"stateId": state_id}
        ),
        "statement_normalization": await client.statementnormalizationtrace.find_many(
            {"stateId": state_id}
        ),
        "autoformalization": await client.autoformalizationtrace.find_many(
            {"stateId": state_id}
        ),
        "semantic_check": await client.semanticchecktrace.find_many(
            {"stateId": state_id}
        ),
        "statement_correction": await client.statementcorrectiontrace.find_many(
            {"stateId": state_id}
        ),
        "statement_refinement": await client.statementrefinementtrace.find_many(
            {"stateId": state_id}
        ),
        "formalization_selection": await client.formalizationselectiontrace.find_many(
            {"stateId": state_id}
        ),
        "subgoal_extraction": await client.subgoalextractiontrace.find_many(
            {"stateId": state_id}
        ),
    }

    # Sort each trace list by creation time
    for key in traces:
        traces[key].sort(key=lambda x: x.get("createdAt", datetime.min))

    return traces


async def fetch_trace_counts_for_state(state_id: str) -> Dict[str, int]:
    """Fetch trace counts for a state without fetching full trace data."""
    client = await get_connected_client()

    counts = {
        "prover": len(await client.provertrace.find_many({"stateId": state_id})),
        "self_correction": len(
            await client.selfcorrectiontrace.find_many({"stateId": state_id})
        ),
        "proof_correction": len(
            await client.proofcorrectiontrace.find_many({"stateId": state_id})
        ),
        "sketch_correction": len(
            await client.sketchcorrectiontrace.find_many({"stateId": state_id})
        ),
        "correctness_check": len(
            await client.correctnesschecktrace.find_many({"stateId": state_id})
        ),
        "shallow_solve": len(
            await client.shallowsolvetrace.find_many({"stateId": state_id})
        ),
        "theorem_retrieval": len(
            await client.theoremretrievaltrace.find_many({"stateId": state_id})
        ),
        "definition_retrieval": len(
            await client.definitionretrievaltrace.find_many({"stateId": state_id})
        ),
        "informal_proof": len(
            await client.informalprooftrace.find_many({"stateId": state_id})
        ),
        "proof_sketch": len(
            await client.proofsketchtrace.find_many({"stateId": state_id})
        ),
        "proof_assembly": len(
            await client.proofassemblytrace.find_many({"stateId": state_id})
        ),
        "statement_normalization": len(
            await client.statementnormalizationtrace.find_many({"stateId": state_id})
        ),
        "autoformalization": len(
            await client.autoformalizationtrace.find_many({"stateId": state_id})
        ),
        "semantic_check": len(
            await client.semanticchecktrace.find_many({"stateId": state_id})
        ),
        "statement_correction": len(
            await client.statementcorrectiontrace.find_many({"stateId": state_id})
        ),
        "statement_refinement": len(
            await client.statementrefinementtrace.find_many({"stateId": state_id})
        ),
        "formalization_selection": len(
            await client.formalizationselectiontrace.find_many({"stateId": state_id})
        ),
        "subgoal_extraction": len(
            await client.subgoalextractiontrace.find_many({"stateId": state_id})
        ),
    }

    return counts


async def fetch_state_by_id(state_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single state by ID."""
    client = await get_connected_client()
    states = await client.theoremforgestate.find_many({"id": state_id})
    return states[0] if states else None


def format_timestamp(ts: Any) -> str:
    """Format timestamp for display."""
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def get_date_range(date_filter: str) -> tuple[datetime, datetime]:
    """Get start and end datetime based on date filter selection."""
    now = datetime.now()

    if date_filter == "Today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif date_filter == "Yesterday":
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif date_filter == "Last 7 days":
        start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif date_filter == "Last 30 days":
        start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif date_filter == "This week":
        # Start from Monday of current week
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif date_filter == "This month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now
    else:  # "All time"
        start = datetime.min
        end = datetime.max

    return start, end


def filter_states_by_date(states: List[Dict[str, Any]], start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Filter states by creation date."""
    filtered = []
    for state in states:
        created_at = state.get("createdAt")
        if isinstance(created_at, datetime):
            if start_date <= created_at <= end_date:
                filtered.append(state)
        # If no timestamp or invalid, include in "All time" view
        elif start_date == datetime.min:
            filtered.append(state)
    return filtered


def display_state_summary(
    state: Dict[str, Any], trace_counts: Optional[Dict[str, int]] = None
):
    """Display a TheoremForgeState as a summary card (for overview page)."""
    state_id = state.get("id", state.get("_id", "Unknown"))

    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(
                f"**ID:** `{state_id[:12]}...`"
                if len(str(state_id)) > 12
                else f"**ID:** `{state_id}`"
            )

            # Display subgoal number if this is a subgoal
            parent_id = state.get("parentId")
            siblings = state.get("siblings", [])
            if parent_id and siblings:
                try:
                    subgoal_index = siblings.index(state_id)
                    subgoal_number = subgoal_index + 1
                    total_subgoals = len(siblings)
                    st.caption(
                        f"🎯 Subgoal {subgoal_number}/{total_subgoals} (Parent: `{parent_id[:8]}...`)"
                    )
                except ValueError:
                    st.caption(f"🎯 Subgoal (Parent: `{parent_id[:8]}...`)")

            st.caption(f"Created: {format_timestamp(state.get('createdAt'))}")

        with col2:
            # Show trace count if available
            if trace_counts:
                total = sum(trace_counts.values())
                st.metric("Total Traces", total)
            else:
                st.write(f"Updated: {format_timestamp(state.get('updatedAt'))}")

        with col3:
            # Use 'success' field from database
            success = state.get("success")
            if success is True:
                st.markdown(
                    '<span class="success-badge">✓ SUCCESS</span>',
                    unsafe_allow_html=True,
                )
            elif success is False:
                st.markdown(
                    '<span class="error-badge">✗ FAILED</span>', unsafe_allow_html=True
                )
            else:
                st.write("**Status:** Unknown")

        # Show truncated problem (using actual field names from database)
        problem = state.get("informalStatement", "") or state.get("formalStatement", "")
        if problem:
            truncated = problem[:200] + "..." if len(problem) > 200 else problem
            st.markdown(f"**Problem:** `{truncated}`")

        # Show if proof exists (using actual field names)
        if "formalProof" in state and state["formalProof"]:
            st.success("✅ Final proof available")

        # Show success status
        if "success" in state:
            if state["success"]:
                st.success("✅ Proof completed successfully")
            else:
                st.warning("⚠️ Proof attempt failed")


def display_state_detail(state: Dict[str, Any]):
    """Display a TheoremForgeState in detail (for detail page)."""
    state_id = state.get("id", state.get("_id", "Unknown"))

    with st.container():
        st.markdown(f"### 🎯 State: `{state_id}`")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.write(f"**Created:** {format_timestamp(state.get('createdAt'))}")
        with col2:
            st.write(f"**Updated:** {format_timestamp(state.get('updatedAt'))}")
        with col3:
            # Use 'success' field from database
            success = state.get("success")
            if success is True:
                st.markdown(
                    '<span class="success-badge">✓ SUCCESS</span>',
                    unsafe_allow_html=True,
                )
            elif success is False:
                st.markdown(
                    '<span class="error-badge">✗ FAILED</span>', unsafe_allow_html=True
                )
            else:
                st.write("**Status:** Unknown")

        # Display main state information (using actual field names from database)
        if "informalStatement" in state and state["informalStatement"]:
            st.markdown("#### 📝 Informal Statement")
            st.markdown(state["informalStatement"])

        if "formalStatement" in state and state["formalStatement"]:
            st.markdown("#### 🎓 Formal Statement")
            st.markdown(f"```lean\n{state['formalStatement']}\n```")

        if "informalProof" in state and state["informalProof"]:
            st.markdown("#### 📖 Informal Proof")
            st.markdown(state["informalProof"])

        if "proofSketch" in state and state["proofSketch"]:
            st.markdown("#### ✏️ Proof Sketch")
            st.markdown(f"```lean\n{state['proofSketch']}\n```")

        if "formalProof" in state and state["formalProof"]:
            st.markdown("#### ✅ Final Formal Proof")
            st.markdown(f"```lean\n{state['formalProof']}\n```")

        # Show success status
        if "success" in state:
            if state["success"]:
                st.success("✓ Proof completed successfully")
            else:
                st.error("✗ Proof attempt failed")

        # Display any additional metadata
        with st.expander("📊 Full State Metadata"):
            display_dict = {
                k: v
                for k, v in state.items()
                if k
                not in [
                    "informalStatement",
                    "formalStatement",
                    "informalProof",
                    "proofSketch",
                    "formalProof",
                    "_id",
                ]
            }
            st.json(display_dict)


def display_trace_section(trace_type: str, traces: List[Dict[str, Any]], emoji: str, current_state: Optional[Dict[str, Any]] = None):
    """Display a section of traces."""
    if not traces:
        return

    st.markdown(
        f"### {emoji} {trace_type.replace('_', ' ').title()} Traces ({len(traces)})"
    )

    for idx, trace in enumerate(traces, 1):
        timestamp = format_timestamp(trace.get("createdAt"))

        # Special title for shallow_solve traces with round number
        if trace_type == "shallow_solve" and "round" in trace:
            round_num = trace["round"]
            total_rounds = trace.get("totalRounds", "?")
            title = f"Trace #{idx} - Round {round_num} (of {total_rounds}) - {timestamp}"
        else:
            title = f"Trace #{idx} - {timestamp}"

        with st.expander(title):
            # Display different fields based on trace type
            if trace_type == "proof_assembly":
                # Special handling for proof assembly traces
                st.markdown("#### 🏗️ Proof Assembly Information")
                
                # Show subgoal IDs if available
                if "subgoalIds" in trace and trace["subgoalIds"]:
                    st.markdown(f"**Number of Subgoals:** {len(trace['subgoalIds'])}")
                    st.markdown("**Subgoal IDs:**")
                    for i, subgoal_id in enumerate(trace["subgoalIds"], 1):
                        st.code(subgoal_id, language=None)
                        
                        # Fetch and display subgoal proof
                        try:
                            subgoal_state = asyncio.run(fetch_state_by_id(subgoal_id))
                            if subgoal_state:
                                st.markdown(f"**Subgoal {i} Details:**")
                                
                                # Show subgoal formal statement
                                if "formalStatement" in subgoal_state and subgoal_state["formalStatement"]:
                                    st.markdown("*Formal Statement:*")
                                    st.code(subgoal_state["formalStatement"], language="lean")
                                
                                # Show subgoal proof
                                if "formalProof" in subgoal_state and subgoal_state["formalProof"]:
                                    st.markdown("*Subgoal Proof:*")
                                    st.code(subgoal_state["formalProof"], language="lean")
                                else:
                                    st.warning(f"⚠️ Subgoal {i} has no formal proof yet")
                                
                                # Show success status
                                if "success" in subgoal_state:
                                    if subgoal_state["success"]:
                                        st.success(f"✅ Subgoal {i} completed successfully")
                                    else:
                                        st.error(f"❌ Subgoal {i} failed")
                            else:
                                st.warning(f"⚠️ Could not fetch subgoal state: {subgoal_id}")
                        except Exception as e:
                            st.error(f"Error fetching subgoal {i}: {str(e)}")
                        
                        st.markdown("---")
                
                # Show assembled proof if available
                if "outputCode" in trace and trace["outputCode"]:
                    st.markdown("#### 🔧 Assembled Proof")
                    st.code(trace["outputCode"], language="lean")
                
                # Show prompt and output
                if "prompt" in trace and trace["prompt"]:
                    st.markdown("#### 📥 Prompt")
                    st.markdown(trace["prompt"])
                
                if "output" in trace and trace["output"]:
                    st.markdown("#### 📤 Output")
                    st.markdown(trace["output"])
                
                # Show validation status
                if "valid" in trace:
                    if trace["valid"]:
                        st.success("✓ Assembly Valid")
                    else:
                        st.error("✗ Assembly Invalid")
                
                # Show error message if exists
                if "errorMessage" in trace and trace["errorMessage"]:
                    st.markdown("#### ⚠️ Error Message")
                    st.error(trace["errorMessage"])
                    
            elif trace_type == "formalization_selection":
                # Special handling for formalization selection traces
                st.markdown("#### 🎯 Formalization Selection")
                
                # Show analysis
                if "analysis" in trace and trace["analysis"]:
                    st.markdown("**Analysis:**")
                    st.markdown(trace["analysis"])
                
                # Show selected index
                if "selectedIndex" in trace:
                    st.markdown(f"**Selected Formalization:** #{trace['selectedIndex']}")
                
                # Show selected formalization
                if "selectedFormalization" in trace and trace["selectedFormalization"]:
                    st.markdown("**Selected Formalization Code:**")
                    st.code(trace["selectedFormalization"], language="lean")
                
                # Show prompt and output
                if "prompt" in trace and trace["prompt"]:
                    st.markdown("#### 📥 Prompt")
                    st.markdown(trace["prompt"])
                
                if "output" in trace and trace["output"]:
                    st.markdown("#### 📤 Output")
                    st.markdown(trace["output"])
                    
            elif trace_type == "subgoal_extraction":
                # Special handling for subgoal extraction traces
                st.markdown("#### 🧩 Subgoal Extraction")
                
                # Show number of subgoals extracted
                if "subgoals" in trace and trace["subgoals"]:
                    st.markdown(f"**Number of Subgoals Extracted:** {len(trace['subgoals'])}")
                    st.markdown("**Extracted Subgoals:**")
                    for i, subgoal in enumerate(trace["subgoals"], 1):
                        st.markdown(f"**Subgoal {i}:**")
                        st.code(subgoal, language="lean")
                
                # Show subgoal IDs if available
                if "subgoalIds" in trace and trace["subgoalIds"]:
                    st.markdown("**Subgoal IDs:**")
                    for i, subgoal_id in enumerate(trace["subgoalIds"], 1):
                        st.markdown(f"{i}. `{subgoal_id}`")
                
                # Show prompt and output
                if "prompt" in trace and trace["prompt"]:
                    st.markdown("#### 📥 Prompt")
                    st.markdown(trace["prompt"])
                
                if "output" in trace and trace["output"]:
                    st.markdown("#### 📤 Output")
                    st.markdown(trace["output"])
                    
            elif trace_type in ["theorem_retrieval", "definition_retrieval"]:
                # Retrieval traces have special structure
                if "queryGenerationPrompt" in trace:
                    st.markdown("#### 📥 Query Generation Prompt")
                    st.markdown(trace["queryGenerationPrompt"])
                if "queryGenerationOutput" in trace:
                    st.markdown("#### 📤 Query Generation Output")
                    st.markdown(trace["queryGenerationOutput"])
                if "queryResults" in trace and trace["queryResults"]:
                    st.markdown("#### 🔍 Query Results")
                    for i, result in enumerate(trace["queryResults"], 1):
                        st.markdown(f"**Result {i}:**")
                        st.code(result, language="lean")
                if "theoremSelectionPrompt" in trace:
                    st.markdown("#### 📥 Selection Prompt")
                    st.markdown(trace["theoremSelectionPrompt"])
                if "definitionSelectionPrompt" in trace:
                    st.markdown("#### 📥 Selection Prompt")
                    st.markdown(trace["definitionSelectionPrompt"])
                if "theoremSelectionOutput" in trace:
                    st.markdown("#### 📤 Selection Output")
                    st.markdown(trace["theoremSelectionOutput"])
                if "definitionSelectionOutput" in trace:
                    st.markdown("#### 📤 Selection Output")
                    st.markdown(trace["definitionSelectionOutput"])
                if (
                    "theoremSelectionResults" in trace
                    and trace["theoremSelectionResults"]
                ):
                    st.markdown("#### ✅ Selected Items")
                    for i, theorem in enumerate(trace["theoremSelectionResults"], 1):
                        st.markdown(f"**Item {i}:**")
                        st.code(theorem, language="lean")
                if (
                    "definitionSelectionResults" in trace
                    and trace["definitionSelectionResults"]
                ):
                    st.markdown("#### ✅ Selected Definitions")
                    for i, definition in enumerate(trace["definitionSelectionResults"], 1):
                        st.markdown(f"**Definition {i}:**")
                        st.code(definition, language="lean")
            else:
                # Standard trace structure with prompt and output
                if "prompt" in trace and trace["prompt"]:
                    st.markdown("#### 📥 Prompt")
                    st.markdown(trace["prompt"])

                if "output" in trace and trace["output"]:
                    st.markdown("#### 📤 Output")
                    st.markdown(trace["output"])

                # Display code output if exists
                if "outputCode" in trace and trace["outputCode"]:
                    st.markdown("#### 💻 Generated Code")
                    st.code(trace["outputCode"], language="lean")

                # Display validation status
                if "valid" in trace:
                    if trace["valid"]:
                        st.success("✓ Valid")
                    else:
                        st.error("✗ Invalid")

                # Display error message if exists
                if "errorMessage" in trace and trace["errorMessage"]:
                    st.markdown("#### ⚠️ Error Message")
                    st.error(trace["errorMessage"])

                # Special display for shallow_solve traces
                if trace_type == "shallow_solve":
                    if "round" in trace:
                        st.markdown(f"**Round:** {trace['round']}")
                    if "totalRounds" in trace:
                        st.markdown(f"**Total Rounds:** {trace['totalRounds']}")

                    # Show progress
                    if "round" in trace and "totalRounds" in trace:
                        progress = (trace["round"] + 1) / trace["totalRounds"]
                        st.progress(progress)

            # Show full trace data
            with st.expander("🔍 Full Trace Data"):
                display_dict = {k: v for k, v in trace.items() if k != "_id"}
                st.json(display_dict)


def show_overview_page(states: List[Dict[str, Any]], all_states: List[Dict[str, Any]]):
    """Display overview of root states only."""
    st.markdown("## 📊 States Overview")

    # Date filter (defaults to "Today")
    st.markdown("### 📅 Date Filter")
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        date_filter = st.selectbox(
            "Select Date Range",
            ["Today", "Yesterday", "Last 7 days", "Last 30 days", "This week", "This month", "All time"],
            index=0  # Default to "Today"
        )
        # Save to session state for sidebar display
        st.session_state.date_filter = date_filter

    # Get date range and apply filter
    start_date, end_date = get_date_range(date_filter)

    # Apply date filter to both root states and all states
    states_filtered_by_date = filter_states_by_date(states, start_date, end_date)
    all_states_filtered_by_date = filter_states_by_date(all_states, start_date, end_date)

    with col2:
        if date_filter == "Today":
            st.info(f"📅 Showing traces from today ({start_date.strftime('%Y-%m-%d')})")
        elif date_filter == "All time":
            st.info("📅 Showing all traces")
        else:
            st.info(f"📅 {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    with col3:
        st.metric("States in Range", len(states_filtered_by_date))

    # Show statistics
    total_states = len(all_states_filtered_by_date)
    root_states_count = len(states_filtered_by_date)
    subgoals_count = total_states - root_states_count

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Root States", root_states_count)
    with col2:
        st.metric("Subgoals Generated", subgoals_count)
    with col3:
        st.metric("Total States", total_states)

    # Add toggle for showing subgoals
    show_subgoals = st.checkbox("📎 Show subgoals in overview", value=False)

    if not show_subgoals:
        st.info(
            "📌 Currently displaying only root states (manually submitted theorems). Enable 'Show subgoals' to view all states."
        )

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status", ["All", "completed", "failed", "in_progress", "pending"]
        )
    with col2:
        sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First"])

    # Choose which states to display based on toggle
    display_states = all_states_filtered_by_date if show_subgoals else states_filtered_by_date

    # Apply status filters (using 'success' field from database)
    filtered_states = display_states
    if status_filter == "completed":
        filtered_states = [s for s in display_states if s.get("success") is True]
    elif status_filter == "failed":
        filtered_states = [s for s in display_states if s.get("success") is False]
    # "All" and other statuses show everything

    if sort_order == "Oldest First":
        filtered_states = list(reversed(filtered_states))

    state_type = "states" if show_subgoals else "root states"
    st.markdown(f"Showing **{len(filtered_states)}** {state_type}")
    st.markdown("---")

    # Display states
    for idx, state in enumerate(filtered_states):
        state_id = state.get("id", state.get("_id", "Unknown"))

        # Build expander title with subgoal information
        title = (
            f"State {idx + 1}: {state_id} - {format_timestamp(state.get('createdAt'))}"
        )

        # Add subgoal count for root states
        if not state.get("parentId"):
            subgoals = state.get("subgoals", [])
            if subgoals:
                title += f" | 📎 {len(subgoals)} subgoal(s)"
        else:
            # Add subgoal number for subgoals
            siblings = state.get("siblings", [])
            if siblings:
                try:
                    subgoal_index = siblings.index(state_id)
                    subgoal_number = subgoal_index + 1
                    total_subgoals = len(siblings)
                    title += f" | 🎯 Subgoal {subgoal_number}/{total_subgoals}"
                except ValueError:
                    title += " | 🎯 Subgoal"

        with st.expander(title, expanded=False):
            display_state_summary(state)

            # Add button to view details
            if st.button("View Details & Traces", key=f"view_{state_id}"):
                st.session_state.view_mode = "detail"
                st.session_state.selected_state_id = state_id
                st.rerun()


def show_detail_page(all_states: List[Dict[str, Any]]):
    """Display detail view for a single state (can be root or subgoal)."""
    # Back button
    if st.button("← Back to Overview"):
        st.session_state.view_mode = "overview"
        st.session_state.selected_state_id = None
        st.rerun()

    selected_state_id = st.session_state.get("selected_state_id")
    if not selected_state_id:
        st.error("No state selected")
        return

    # Find the selected state from all states
    selected_state = None
    for state in all_states:
        if state.get("id", state.get("_id")) == selected_state_id:
            selected_state = state
            break

    if not selected_state:
        st.error(f"State {selected_state_id} not found")
        return

    # Show if this is a subgoal
    if selected_state.get("parentId"):
        st.info(f"📎 This is a subgoal. Parent ID: `{selected_state['parentId']}`")

    # Trace type filters
    st.sidebar.markdown("### 🔍 Trace Filters")

    # Pre-processing stage
    st.sidebar.markdown("**Pre-processing:**")
    show_statement_normalization = st.sidebar.checkbox("Statement Normalization", value=True)
    show_autoformalization = st.sidebar.checkbox("Autoformalization", value=True)
    show_semantic_check = st.sidebar.checkbox("Semantic Check", value=True)
    show_statement_correction = st.sidebar.checkbox("Statement Correction", value=True)
    show_statement_refinement = st.sidebar.checkbox("Statement Refinement", value=True)
    show_formalization_selection = st.sidebar.checkbox("Formalization Selection", value=True)

    # Retrieval stage
    st.sidebar.markdown("**Retrieval:**")
    show_theorem_retrieval = st.sidebar.checkbox("Theorem Retrieval", value=True)
    show_definition_retrieval = st.sidebar.checkbox("Definition Retrieval", value=True)

    # Planning stage
    st.sidebar.markdown("**Planning:**")
    show_informal_proof = st.sidebar.checkbox("Informal Proof", value=True)
    show_proof_sketch = st.sidebar.checkbox("Proof Sketch", value=True)
    show_subgoal_extraction = st.sidebar.checkbox("Subgoal Extraction", value=True)

    # Proving stage
    st.sidebar.markdown("**Proving:**")
    show_prover = st.sidebar.checkbox("Prover", value=True)
    show_proof_correction = st.sidebar.checkbox("Proof Correction", value=True)
    show_sketch_correction = st.sidebar.checkbox("Sketch Correction", value=True)
    show_correctness_check = st.sidebar.checkbox("Correctness Check", value=True)
    show_shallow_solve = st.sidebar.checkbox("Shallow Solve", value=True)
    show_proof_assembly = st.sidebar.checkbox("Proof Assembly", value=True)

    # Legacy
    st.sidebar.markdown("**Legacy:**")
    show_self_correction = st.sidebar.checkbox("Self-Correction (deprecated)", value=False)

    # Display selected state
    st.markdown("---")
    display_state_detail(selected_state)

    # Fetch and display traces
    st.markdown("---")
    st.markdown("## 📋 Execution Traces")

    try:
        # Use asyncio.run to create a fresh event loop for each fetch
        traces = asyncio.run(fetch_traces_for_state(selected_state_id))
    except Exception as e:
        st.error(f"Failed to fetch traces: {e}")
        st.code(f"Error details: {str(e)}")
        return

    # Display traces based on filters (in chronological order of pipeline)

    # Pre-processing stage
    if show_statement_normalization:
        display_trace_section("statement_normalization", traces["statement_normalization"], "📝", selected_state)

    if show_autoformalization:
        display_trace_section("autoformalization", traces["autoformalization"], "🔄", selected_state)

    if show_semantic_check:
        display_trace_section("semantic_check", traces["semantic_check"], "✅", selected_state)

    if show_statement_correction:
        display_trace_section("statement_correction", traces["statement_correction"], "🔧", selected_state)

    if show_statement_refinement:
        display_trace_section("statement_refinement", traces["statement_refinement"], "✨", selected_state)

    if show_formalization_selection:
        display_trace_section("formalization_selection", traces["formalization_selection"], "🎯", selected_state)

    # Retrieval stage
    if show_theorem_retrieval:
        display_trace_section("theorem_retrieval", traces["theorem_retrieval"], "🔍", selected_state)

    if show_definition_retrieval:
        display_trace_section("definition_retrieval", traces["definition_retrieval"], "📖", selected_state)

    # Planning stage
    if show_informal_proof:
        display_trace_section("informal_proof", traces["informal_proof"], "💭", selected_state)

    if show_proof_sketch:
        display_trace_section("proof_sketch", traces["proof_sketch"], "✏️", selected_state)

    if show_subgoal_extraction:
        display_trace_section("subgoal_extraction", traces["subgoal_extraction"], "🧩", selected_state)

    # Proving stage
    if show_prover:
        display_trace_section("prover", traces["prover"], "🤖", selected_state)

    if show_proof_correction:
        display_trace_section("proof_correction", traces["proof_correction"], "🔧", selected_state)

    if show_sketch_correction:
        display_trace_section("sketch_correction", traces["sketch_correction"], "🛠️", selected_state)

    if show_correctness_check:
        display_trace_section("correctness_check", traces["correctness_check"], "🎯", selected_state)

    if show_shallow_solve:
        display_trace_section("shallow_solve", traces["shallow_solve"], "🔄", selected_state)

    if show_proof_assembly:
        display_trace_section("proof_assembly", traces["proof_assembly"], "🏗️", selected_state)

    # Legacy
    if show_self_correction:
        display_trace_section("self_correction", traces["self_correction"], "♻️", selected_state)

    # Summary statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Trace Statistics")

    # Pre-processing
    st.sidebar.markdown("**Pre-processing:**")
    st.sidebar.write(f"Statement Norm: {len(traces['statement_normalization'])}")
    st.sidebar.write(f"Autoformalization: {len(traces['autoformalization'])}")
    st.sidebar.write(f"Semantic Check: {len(traces['semantic_check'])}")
    st.sidebar.write(f"Statement Correction: {len(traces['statement_correction'])}")
    st.sidebar.write(f"Statement Refinement: {len(traces['statement_refinement'])}")
    st.sidebar.write(f"Formalization Select: {len(traces['formalization_selection'])}")

    # Retrieval
    st.sidebar.markdown("**Retrieval:**")
    st.sidebar.write(f"Theorem: {len(traces['theorem_retrieval'])}")
    st.sidebar.write(f"Definition: {len(traces['definition_retrieval'])}")

    # Planning
    st.sidebar.markdown("**Planning:**")
    st.sidebar.write(f"Informal Proof: {len(traces['informal_proof'])}")
    st.sidebar.write(f"Proof Sketch: {len(traces['proof_sketch'])}")
    st.sidebar.write(f"Subgoal Extraction: {len(traces['subgoal_extraction'])}")

    # Proving
    st.sidebar.markdown("**Proving:**")
    st.sidebar.write(f"Prover: {len(traces['prover'])}")
    st.sidebar.write(f"Proof Correction: {len(traces['proof_correction'])}")
    st.sidebar.write(f"Sketch Correction: {len(traces['sketch_correction'])}")
    st.sidebar.write(f"Correctness Check: {len(traces['correctness_check'])}")
    st.sidebar.write(f"Shallow Solve: {len(traces['shallow_solve'])}")
    st.sidebar.write(f"Proof Assembly: {len(traces['proof_assembly'])}")

    # Legacy
    if len(traces['self_correction']) > 0:
        st.sidebar.markdown("**Legacy:**")
        st.sidebar.write(f"Self-Correction: {len(traces['self_correction'])}")

    # Total
    st.sidebar.markdown("---")
    total_traces = sum(len(v) for v in traces.values())
    st.sidebar.write(f"**Total Traces: {total_traces}**")


def main():
    """Main Streamlit app."""
    st.markdown(
        '<h1 class="main-header">📐 TheoremForge Trace Visualization</h1>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    This dashboard displays TheoremForge states and their associated traces.
    Each state represents a proof attempt with detailed execution traces.
    """)

    # Initialize session state
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "overview"
    if "selected_state_id" not in st.session_state:
        st.session_state.selected_state_id = None
    if "date_filter" not in st.session_state:
        st.session_state.date_filter = "Today"

    # Sidebar controls
    st.sidebar.title("🎛️ Controls")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()

    # Fetch states with better error handling
    try:
        # Use asyncio.run to create a fresh event loop for each fetch
        all_states = asyncio.run(fetch_states())
        root_states = asyncio.run(fetch_root_states())
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info(
            "Make sure MongoDB is running and DATABASE_URL is configured correctly."
        )
        st.code(f"Error details: {str(e)}")
        return

    if not all_states:
        st.warning("No states found in the database.")
        st.info("Run TheoremForge to generate states and traces.")
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Database Statistics")
    st.sidebar.markdown(f"**Total States:** {len(all_states)}")
    st.sidebar.markdown(f"**Root States:** {len(root_states)}")
    st.sidebar.markdown(f"**Subgoals:** {len(all_states) - len(root_states)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Current View")
    st.sidebar.markdown(f"**View Mode:** {st.session_state.view_mode.title()}")

    # Show date filter info only in overview mode
    if st.session_state.view_mode == "overview":
        date_filter = st.session_state.get("date_filter", "Today")
        st.sidebar.markdown(f"**Date Filter:** {date_filter}")

    # Show appropriate page based on view mode
    if st.session_state.view_mode == "overview":
        show_overview_page(root_states, all_states)
    else:
        show_detail_page(all_states)


if __name__ == "__main__":
    main()
