"""
Streamlit Dashboard for Multi-Agent Coverage Visualization.

Interactive visualization of agent exploration, communication, and coverage.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/Users/srimaan/.openclaw/workspaces/repo-cce')

# Page config
st.set_page_config(
    page_title="FAI Multi-Agent Coverage Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stSlider > div > div > div > div {
        background: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_trajectory(filepath: str):
    """Load trajectory from pickle file."""
    with open(filepath, 'rb') as f:
        trajectory = pickle.load(f)
    return trajectory


def create_grid_figure(data: dict, step_idx: int) -> go.Figure:
    """
    Create layered Plotly figure for the grid visualization.
    
    Layers:
    1. Base grid (obstacles = black, free = light gray)
    2. Visited overlay (colored by agent ID, semi-transparent)
    3. Agent positions (star markers)
    4. Communication lines (dashed lines between sender/receiver)
    """
    grid = data['grid']
    visited_global = data['visited_global']
    agent_positions = data['agent_positions']
    messages = data['messages']
    grid_size = grid.shape[0]
    
    fig = go.Figure()
    
    # Layer 1: Base grid
    # Create custom colorscale: 0=light gray (free), 1=black (obstacle)
    base_grid = grid.copy()
    fig.add_trace(go.Heatmap(
        z=base_grid,
        colorscale=[[0, '#f1f5f9'], [1, '#1e293b']],  # Light gray to black
        showscale=False,
        opacity=1.0,
        hoverinfo='skip',
        zmin=0,
        zmax=1
    ))
    
    # Layer 2: Visited overlay (colored by agent ID)
    # Create a discrete colormap for agents
    num_agents = len(agent_positions)
    # Generate distinct colors for each agent
    agent_colors = [
        '#ef4444',  # Red - Agent 1
        '#3b82f6',  # Blue - Agent 2
        '#22c55e',  # Green - Agent 3
        '#f59e0b',  # Yellow - Agent 4
        '#a855f7',  # Purple - Agent 5
        '#ec4899',  # Pink - Agent 6
    ]
    
    # Create visited overlay with agent colors
    visited_overlay = np.full((grid_size, grid_size, 4), 255, dtype=np.uint8)
    for agent_id in range(1, num_agents + 1):
        mask = visited_global == agent_id
        if mask.any():
            color_hex = agent_colors[(agent_id - 1) % len(agent_colors)]
            # Convert hex to RGBA
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            visited_overlay[mask] = [r, g, b, 180]  # 0.7 opacity
    
    # Add visited overlay as heatmap with custom colors
    # We'll use a numeric representation with custom colorscale
    visited_numeric = visited_global.astype(float)
    visited_numeric[visited_numeric == 0] = np.nan  # Unvisited = transparent
    
    # Create colorscale for visited cells
    colorscale = [[0, 'rgba(0,0,0,0)']]  # Unvisited = transparent
    for i in range(1, num_agents + 1):
        colorscale.append([i / num_agents, agent_colors[(i - 1) % len(agent_colors)]])
    
    fig.add_trace(go.Heatmap(
        z=visited_numeric,
        colorscale=colorscale,
        showscale=False,
        opacity=0.7,
        hoverinfo='skip',
        zmin=1,
        zmax=num_agents
    ))
    
    # Layer 3: Agent positions (star markers)
    for i, (y, x) in enumerate(agent_positions):
        color = agent_colors[i % len(agent_colors)]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(
                symbol='star',
                size=20,
                color=color,
                line=dict(width=2, color='white')
            ),
            name=f'Agent {i+1}',
            showlegend=True,
            hovertemplate=f'Agent {i+1}<br>Position: ({y}, {x})<extra></extra>'
        ))
    
    # Layer 4: Communication lines (dashed lines for current step messages)
    for msg in messages:
        sender = msg['sender']
        receiver = msg['receiver']
        if sender < len(agent_positions) and receiver < len(agent_positions):
            sy, sx = agent_positions[sender]
            ry, rx = agent_positions[receiver]
            
            sender_color = agent_colors[sender % len(agent_colors)]
            fig.add_trace(go.Scatter(
                x=[sx, rx],
                y=[sy, ry],
                mode='lines',
                line=dict(
                    color=sender_color,
                    width=3,
                    dash='dash'
                ),
                showlegend=False,
                hovertemplate=f'Message: Agent {sender+1} → Agent {receiver+1}<br>Code: {msg.get("code", "N/A")}<br>Utility: {msg.get("utility", 0):.2f}<extra></extra>'
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"Timestep {step_idx}",
            font=dict(size=20, color='#1f2937'),
            x=0.5
        ),
        xaxis=dict(
            range=[-0.5, grid_size - 0.5],
            showgrid=True,
            gridcolor='#e2e8f0',
            gridwidth=1,
            tickmode='linear',
            dtick=1,
            showticklabels=True,
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            range=[grid_size - 0.5, -0.5],  # Invert Y so (0,0) is top-left
            showgrid=True,
            gridcolor='#e2e8f0',
            gridwidth=1,
            tickmode='linear',
            dtick=1,
            showticklabels=True
        ),
        width=700,
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_coverage_chart(history: list, current_step: int) -> go.Figure:
    """Create coverage line chart over time."""
    fig = go.Figure()
    
    # Full history
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=history,
        mode='lines',
        line=dict(color='#3b82f6', width=3),
        name='Coverage',
        hovertemplate='Step %{x}<br>Coverage: %{y:.2%}<extra></extra>'
    ))
    
    # Current step marker
    fig.add_trace(go.Scatter(
        x=[current_step],
        y=[history[current_step] if current_step < len(history) else history[-1]],
        mode='markers',
        marker=dict(size=12, color='#ef4444', symbol='circle'),
        name='Current Step',
        showlegend=False
    ))
    
    # 95% target line
    fig.add_trace(go.Scatter(
        x=[0, len(history) - 1],
        y=[0.95, 0.95],
        mode='lines',
        line=dict(color='#22c55e', width=2, dash='dash'),
        name='95% Target',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text="Coverage Over Time",
            font=dict(size=16, color='#1f2937')
        ),
        xaxis=dict(
            title="Timestep",
            showgrid=True,
            gridcolor='#e2e8f0'
        ),
        yaxis=dict(
            title="Coverage",
            tickformat='.0%',
            range=[0, 1.05],
            showgrid=True,
            gridcolor='#e2e8f0'
        ),
        height=350,
        margin=dict(l=50, r=20, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig


def main():
    st.markdown('<div class="main-header">🤖 FAI Multi-Agent Coverage Visualizer</div>', unsafe_allow_html=True)
    st.markdown("Watch agents explore, communicate, and achieve coverage in real-time.")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Load Trajectory")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload trajectory.pkl",
            type=['pkl'],
            help="Select a recorded trajectory file"
        )
        
        # Default file path
        default_path = Path("runs/viz/trajectory.pkl")
        if default_path.exists() and not uploaded_file:
            st.info(f"📄 Found default: `{default_path}`")
            if st.button("Load Default"):
                uploaded_file = str(default_path)
        
        if uploaded_file:
            if isinstance(uploaded_file, str):
                trajectory = load_trajectory(uploaded_file)
            else:
                trajectory = pickle.load(uploaded_file)
            
            st.success(f"✅ Loaded {len(trajectory)} timesteps")
            
            # Timestep slider
            step_idx = st.slider(
                "Timestep",
                min_value=0,
                max_value=len(trajectory) - 1,
                value=0,
                help="Drag to scrub through the episode"
            )
            
            # Play button
            if st.button("▶️ Play Episode"):
                # Auto-play through timesteps
                placeholder = st.empty()
                for i in range(len(trajectory)):
                    placeholder.empty()
                    with placeholder.container():
                        step_idx = i
                        # Re-render (Streamlit limitation - just show final)
                st.rerun()
            
            # Show current step data
            record = trajectory[step_idx]
            data = record.get_data_for_step(step_idx) if hasattr(record, 'get_data_for_step') else record
            
            # Display data
            st.divider()
            st.subheader("📊 Current Step Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coverage", f"{data['coverage']:.1%}")
            with col2:
                st.metric("Bandwidth Used", f"{data['bandwidth_used']:.0f} bits")
            
            if 'lam' in data:
                st.metric("λ (Efficiency)", f"{data['lam']:.4f}")
            
            # Messages table
            if data['messages']:
                st.subheader("📨 Messages at This Step")
                msg_df = pd.DataFrame(data['messages'])
                if not msg_df.empty:
                    msg_df['sender'] = msg_df['sender'].apply(lambda x: f"Agent {x+1}")
                    msg_df['receiver'] = msg_df['receiver'].apply(lambda x: f"Agent {x+1}")
                    st.dataframe(msg_df, use_container_width=True, hide_index=True)
            else:
                st.info("No messages sent at this step")
        else:
            st.warning("⬆️ Upload a trajectory.pkl file to begin")
            st.stop()
    
    # Main content
    if not uploaded_file:
        return
    
    # Get current step data
    record = trajectory[step_idx]
    data = record.get_data_for_step(step_idx) if hasattr(record, 'get_data_for_step') else record
    
    # Main grid visualization
    col_grid, col_charts = st.columns([2, 1])
    
    with col_grid:
        st.subheader("🗺️ Grid World")
        fig = create_grid_figure(data, step_idx)
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.caption("""
        **Legend:** 
        ⬛ Obstacles · ⬜ Unknown · 🟥🟦🟩🟨 Visited by Agent 1/2/3/4 · ⭐ Agent Position · ➖ Communication
        """)
    
    with col_charts:
        # Coverage chart
        coverage_history = [r.metrics['coverage'] for r in trajectory] if hasattr(trajectory[0], 'metrics') else []
        if coverage_history:
            st.plotly_chart(create_coverage_chart(coverage_history, step_idx), use_container_width=True)
        
        # Additional metrics
        st.subheader("📈 Episode Progress")
        progress = (step_idx + 1) / len(trajectory) * 100
        st.progress(progress / 100)
        st.caption(f"Step {step_idx + 1} of {len(trajectory)} ({progress:.0f}%)")
        
        # Bandwidth history
        if hasattr(trajectory[0], 'metrics'):
            bw_history = [r.metrics['bandwidth_used'] for r in trajectory]
            bw_fig = go.Figure()
            bw_fig.add_trace(go.Scatter(
                x=list(range(len(bw_history))),
                y=bw_history,
                mode='lines',
                line=dict(color='#f59e0b', width=2),
                name='Bandwidth'
            ))
            bw_fig.add_trace(go.Scatter(
                x=[step_idx],
                y=[bw_history[step_idx]],
                mode='markers',
                marker=dict(size=10, color='#ef4444'),
                showlegend=False
            ))
            bw_fig.update_layout(
                title="Bandwidth Used",
                xaxis_title="Timestep",
                yaxis_title="Bits",
                height=200,
                margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(bw_fig, use_container_width=True)
    
    # Full message log (expandable)
    with st.expander("📜 Full Message Log (All Steps)"):
        all_messages = []
        for i, record in enumerate(trajectory):
            step_data = record.get_data_for_step(i) if hasattr(record, 'get_data_for_step') else record
            for msg in step_data['messages']:
                msg_copy = msg.copy()
                msg_copy['step'] = i
                msg_copy['sender'] = f"Agent {msg_copy['sender']+1}"
                msg_copy['receiver'] = f"Agent {msg_copy['receiver']+1}"
                all_messages.append(msg_copy)
        
        if all_messages:
            msg_df = pd.DataFrame(all_messages)
            st.dataframe(msg_df, use_container_width=True, hide_index=True)
        else:
            st.info("No messages sent in this episode")


if __name__ == "__main__":
    main()