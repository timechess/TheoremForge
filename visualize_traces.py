"""
Streamlit visualization for TheoremForge database.

This script displays TheoremForgeState documents as the root and their associated traces.
"""

import asyncio
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
from theoremforge.db import MongoDBClient


# Configure Streamlit page
st.set_page_config(
    page_title="TheoremForge Trace Visualization",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)


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
    states.sort(key=lambda x: x.get('createdAt', datetime.min), reverse=True)
    return states


async def fetch_root_states() -> List[Dict[str, Any]]:
    """Fetch only root TheoremForgeState documents (no parent)."""
    client = await get_connected_client()

    # Query for states where parentId is None or doesn't exist
    states = await client.theoremforgestate.find_many({})
    # Filter for root states (no parent_id)
    root_states = [s for s in states if not s.get('parentId')]
    # Sort by creation time, most recent first
    root_states.sort(key=lambda x: x.get('createdAt', datetime.min), reverse=True)
    return root_states


async def fetch_traces_for_state(state_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all traces associated with a state."""
    client = await get_connected_client()

    # Note: Using 'stateId' to match the database field name
    traces = {
        'prover': await client.provertrace.find_many({'stateId': state_id}),
        'self_correction': await client.selfcorrectiontrace.find_many({'stateId': state_id}),
        'theorem_retrieval': await client.theoremretrievaltrace.find_many({'stateId': state_id}),
        'informal_proof': await client.informalprooftrace.find_many({'stateId': state_id}),
        'proof_sketch': await client.proofsketchtrace.find_many({'stateId': state_id}),
        'proof_assembly': await client.proofassemblytrace.find_many({'stateId': state_id}),
    }

    # Sort each trace list by creation time
    for key in traces:
        traces[key].sort(key=lambda x: x.get('createdAt', datetime.min))

    return traces


async def fetch_trace_counts_for_state(state_id: str) -> Dict[str, int]:
    """Fetch trace counts for a state without fetching full trace data."""
    client = await get_connected_client()

    counts = {
        'prover': len(await client.provertrace.find_many({'stateId': state_id})),
        'self_correction': len(await client.selfcorrectiontrace.find_many({'stateId': state_id})),
        'theorem_retrieval': len(await client.theoremretrievaltrace.find_many({'stateId': state_id})),
        'informal_proof': len(await client.informalprooftrace.find_many({'stateId': state_id})),
        'proof_sketch': len(await client.proofsketchtrace.find_many({'stateId': state_id})),
        'proof_assembly': len(await client.proofassemblytrace.find_many({'stateId': state_id})),
    }

    return counts


def format_timestamp(ts: Any) -> str:
    """Format timestamp for display."""
    if isinstance(ts, datetime):
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    return str(ts)


def display_state_summary(state: Dict[str, Any], trace_counts: Optional[Dict[str, int]] = None):
    """Display a TheoremForgeState as a summary card (for overview page)."""
    state_id = state.get('id', state.get('_id', 'Unknown'))

    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"**ID:** `{state_id[:12]}...`" if len(str(state_id)) > 12 else f"**ID:** `{state_id}`")
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
            success = state.get('success')
            if success is True:
                st.markdown('<span class="success-badge">✓ SUCCESS</span>', unsafe_allow_html=True)
            elif success is False:
                st.markdown('<span class="error-badge">✗ FAILED</span>', unsafe_allow_html=True)
            else:
                st.write("**Status:** Unknown")

        # Show truncated problem (using actual field names from database)
        problem = state.get('informalStatement', '') or state.get('formalStatement', '')
        if problem:
            truncated = problem[:200] + "..." if len(problem) > 200 else problem
            st.markdown(f"**Problem:** `{truncated}`")

        # Show if proof exists (using actual field names)
        if 'formalProof' in state and state['formalProof']:
            st.success("✅ Final proof available")

        # Show success status
        if 'success' in state:
            if state['success']:
                st.success("✅ Proof completed successfully")
            else:
                st.warning("⚠️ Proof attempt failed")


def display_state_detail(state: Dict[str, Any]):
    """Display a TheoremForgeState in detail (for detail page)."""
    state_id = state.get('id', state.get('_id', 'Unknown'))

    with st.container():
        st.markdown(f"### 🎯 State: `{state_id}`")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.write(f"**Created:** {format_timestamp(state.get('createdAt'))}")
        with col2:
            st.write(f"**Updated:** {format_timestamp(state.get('updatedAt'))}")
        with col3:
            # Use 'success' field from database
            success = state.get('success')
            if success is True:
                st.markdown('<span class="success-badge">✓ SUCCESS</span>', unsafe_allow_html=True)
            elif success is False:
                st.markdown('<span class="error-badge">✗ FAILED</span>', unsafe_allow_html=True)
            else:
                st.write("**Status:** Unknown")

        # Display main state information (using actual field names from database)
        if 'informalStatement' in state and state['informalStatement']:
            st.markdown("#### 📝 Informal Statement")
            st.markdown(state['informalStatement'])

        if 'formalStatement' in state and state['formalStatement']:
            st.markdown("#### 🎓 Formal Statement")
            st.markdown(f"```lean\n{state['formalStatement']}\n```")

        if 'informalProof' in state and state['informalProof']:
            st.markdown("#### 📖 Informal Proof")
            st.markdown(state['informalProof'])

        if 'proofSketch' in state and state['proofSketch']:
            st.markdown("#### ✏️ Proof Sketch")
            st.markdown(f"```lean\n{state['proofSketch']}\n```")

        if 'formalProof' in state and state['formalProof']:
            st.markdown("#### ✅ Final Formal Proof")
            st.markdown(f"```lean\n{state['formalProof']}\n```")

        # Show success status
        if 'success' in state:
            if state['success']:
                st.success("✓ Proof completed successfully")
            else:
                st.error("✗ Proof attempt failed")

        # Display any additional metadata
        with st.expander("📊 Full State Metadata"):
            display_dict = {k: v for k, v in state.items()
                          if k not in ['informalStatement', 'formalStatement', 'informalProof',
                                      'proofSketch', 'formalProof', '_id']}
            st.json(display_dict)


def display_trace_section(trace_type: str, traces: List[Dict[str, Any]], emoji: str):
    """Display a section of traces."""
    if not traces:
        return

    st.markdown(f"### {emoji} {trace_type.replace('_', ' ').title()} Traces ({len(traces)})")

    for idx, trace in enumerate(traces, 1):
        timestamp = format_timestamp(trace.get('createdAt'))
        with st.expander(f"Trace #{idx} - {timestamp}"):
            # Display different fields based on trace type
            if trace_type == 'theorem_retrieval':
                # Theorem retrieval has special structure
                if 'queryGenerationPrompt' in trace:
                    st.markdown("#### 📥 Query Generation Prompt")
                    st.markdown(trace['queryGenerationPrompt'])
                if 'queryGenerationOutput' in trace:
                    st.markdown("#### 📤 Query Generation Output")
                    st.markdown(trace['queryGenerationOutput'])
                if 'queryResults' in trace and trace['queryResults']:
                    st.markdown("#### 🔍 Query Results")
                    for i, result in enumerate(trace['queryResults'], 1):
                        st.markdown(f"**Result {i}:**")
                        st.code(result, language='lean')
                if 'theoremSelectionPrompt' in trace:
                    st.markdown("#### 📥 Theorem Selection Prompt")
                    st.markdown(trace['theoremSelectionPrompt'])
                if 'theoremSelectionOutput' in trace:
                    st.markdown("#### 📤 Theorem Selection Output")
                    st.markdown(trace['theoremSelectionOutput'])
                if 'theoremSelectionResults' in trace and trace['theoremSelectionResults']:
                    st.markdown("#### ✅ Selected Theorems")
                    for i, theorem in enumerate(trace['theoremSelectionResults'], 1):
                        st.markdown(f"**Theorem {i}:**")
                        st.code(theorem, language='lean')
            else:
                # Standard trace structure with prompt and output
                if 'prompt' in trace and trace['prompt']:
                    st.markdown("#### 📥 Prompt")
                    st.markdown(trace['prompt'])

                if 'output' in trace and trace['output']:
                    st.markdown("#### 📤 Output")
                    st.markdown(trace['output'])

                # Display code output if exists
                if 'outputCode' in trace and trace['outputCode']:
                    st.markdown("#### 💻 Generated Code")
                    st.code(trace['outputCode'], language='lean')

                # Display validation status
                if 'valid' in trace:
                    if trace['valid']:
                        st.success("✓ Valid")
                    else:
                        st.error("✗ Invalid")

                # Display error message if exists
                if 'errorMessage' in trace and trace['errorMessage']:
                    st.markdown("#### ⚠️ Error Message")
                    st.error(trace['errorMessage'])

            # Show full trace data
            with st.expander("🔍 Full Trace Data"):
                display_dict = {k: v for k, v in trace.items() if k != '_id'}
                st.json(display_dict)


def show_overview_page(states: List[Dict[str, Any]], all_states: List[Dict[str, Any]]):
    """Display overview of root states only."""
    st.markdown("## 📊 Root States Overview")

    # Show statistics
    total_states = len(all_states)
    root_states_count = len(states)
    subgoals_count = total_states - root_states_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Root States", root_states_count)
    with col2:
        st.metric("Subgoals Generated", subgoals_count)
    with col3:
        st.metric("Total States", total_states)

    st.info("📌 This page displays only root states (manually submitted theorems). Subgoals are not shown in the overview.")

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "completed", "failed", "in_progress", "pending"]
        )
    with col2:
        sort_order = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First"]
        )

    # Apply filters (using 'success' field from database)
    filtered_states = states
    if status_filter == "completed":
        filtered_states = [s for s in states if s.get('success') is True]
    elif status_filter == "failed":
        filtered_states = [s for s in states if s.get('success') is False]
    # "All" and other statuses show everything

    if sort_order == "Oldest First":
        filtered_states = list(reversed(filtered_states))

    st.markdown(f"Showing **{len(filtered_states)}** root states")
    st.markdown("---")

    # Display states
    for idx, state in enumerate(filtered_states):
        state_id = state.get('id', state.get('_id', 'Unknown'))

        with st.expander(f"State {idx + 1}: {state_id} - {format_timestamp(state.get('createdAt'))}", expanded=False):
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

    selected_state_id = st.session_state.get('selected_state_id')
    if not selected_state_id:
        st.error("No state selected")
        return

    # Find the selected state from all states
    selected_state = None
    for state in all_states:
        if state.get('id', state.get('_id')) == selected_state_id:
            selected_state = state
            break

    if not selected_state:
        st.error(f"State {selected_state_id} not found")
        return

    # Show if this is a subgoal
    if selected_state.get('parentId'):
        st.info(f"📎 This is a subgoal. Parent ID: `{selected_state['parentId']}`")

    # Trace type filters
    st.sidebar.markdown("### 🔍 Trace Filters")
    show_prover = st.sidebar.checkbox("Prover Traces", value=True)
    show_self_correction = st.sidebar.checkbox("Self-Correction Traces", value=True)
    show_theorem_retrieval = st.sidebar.checkbox("Theorem Retrieval Traces", value=True)
    show_informal_proof = st.sidebar.checkbox("Informal Proof Traces", value=True)
    show_proof_sketch = st.sidebar.checkbox("Proof Sketch Traces", value=True)
    show_proof_assembly = st.sidebar.checkbox("Proof Assembly Traces", value=True)

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

    # Display traces based on filters
    if show_theorem_retrieval:
        display_trace_section("theorem_retrieval", traces['theorem_retrieval'], "🔍")

    if show_informal_proof:
        display_trace_section("informal_proof", traces['informal_proof'], "📝")

    if show_proof_sketch:
        display_trace_section("proof_sketch", traces['proof_sketch'], "✏️")

    if show_proof_assembly:
        display_trace_section("proof_assembly", traces['proof_assembly'], "🔧")

    if show_prover:
        display_trace_section("prover", traces['prover'], "🤖")

    if show_self_correction:
        display_trace_section("self_correction", traces['self_correction'], "🔄")

    # Summary statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Trace Statistics")
    st.sidebar.write(f"Prover: {len(traces['prover'])}")
    st.sidebar.write(f"Self-Correction: {len(traces['self_correction'])}")
    st.sidebar.write(f"Theorem Retrieval: {len(traces['theorem_retrieval'])}")
    st.sidebar.write(f"Informal Proof: {len(traces['informal_proof'])}")
    st.sidebar.write(f"Proof Sketch: {len(traces['proof_sketch'])}")
    st.sidebar.write(f"Proof Assembly: {len(traces['proof_assembly'])}")
    total_traces = sum(len(v) for v in traces.values())
    st.sidebar.write(f"**Total: {total_traces}**")


def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">📐 TheoremForge Trace Visualization</h1>', unsafe_allow_html=True)

    st.markdown("""
    This dashboard displays TheoremForge states and their associated traces.
    Each state represents a proof attempt with detailed execution traces.
    """)

    # Initialize session state
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "overview"
    if 'selected_state_id' not in st.session_state:
        st.session_state.selected_state_id = None

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
        st.info("Make sure MongoDB is running and DATABASE_URL is configured correctly.")
        st.code(f"Error details: {str(e)}")
        return

    if not all_states:
        st.warning("No states found in the database.")
        st.info("Run TheoremForge to generate states and traces.")
        return

    st.sidebar.markdown(f"**Total States:** {len(all_states)}")
    st.sidebar.markdown(f"**Root States:** {len(root_states)}")
    st.sidebar.markdown(f"**Subgoals:** {len(all_states) - len(root_states)}")
    st.sidebar.markdown(f"**Current View:** {st.session_state.view_mode.title()}")

    # Show appropriate page based on view mode
    if st.session_state.view_mode == "overview":
        show_overview_page(root_states, all_states)
    else:
        show_detail_page(all_states)


if __name__ == "__main__":
    main()

