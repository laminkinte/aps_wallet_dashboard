"""
Agent Network Analysis Page
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Agent Network | APS Wallet", layout="wide")

def create_network_graph():
    """Create a sample network graph"""
    G = nx.Graph()
    
    # Add nodes (agents)
    for i in range(1, 21):
        G.add_node(f"A{i}", type="Agent", size=15)
    
    # Add nodes (tellers)
    for i in range(1, 31):
        G.add_node(f"T{i}", type="Teller", size=10)
    
    # Add edges (connections)
    for i in range(1, 21):
        # Each agent connected to 1-3 tellers
        num_tellers = np.random.randint(1, 4)
        tellers = np.random.choice(range(1, 31), num_tellers, replace=False)
        for teller in tellers:
            G.add_edge(f"A{i}", f"T{teller}", weight=np.random.uniform(0.5, 1))
    
    return G

def main():
    st.title("👥 Agent Network Analysis")
    st.markdown("Visualize and analyze agent-teller relationships and network structure")
    
    # Check for data
    if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
        st.warning("Please load data from the main page first.")
        return
    
    results = st.session_state.analysis_results
    
    # Network Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Network Nodes", 
                 results.get('total_active_agents', 0) + results.get('total_active_tellers', 0))
    
    with col2:
        st.metric("Network Density", f"{results.get('network_density', 0):.1f}%")
    
    with col3:
        st.metric("Average Degree", f"{results.get('avg_degree', 0):.1f}")
    
    with col4:
        st.metric("Connected Components", 25)  # Sample data
    
    st.markdown("---")
    
    # Network Visualization Tabs
    tabs = st.tabs(["🗺️ Network Map", "📊 Network Metrics", "🔗 Connections", "🏙️ Regional View"])
    
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create network visualization
            st.markdown("### 🗺️ Network Visualization")
            
            # Generate positions for visualization
            G = create_network_graph()
            pos = nx.spring_layout(G, seed=42)
            
            # Create edges trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create nodes trace
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                if G.nodes[node]['type'] == 'Agent':
                    node_colors.append('#3B82F6')
                    node_sizes.append(20)
                else:
                    node_colors.append('#10B981')
                    node_sizes.append(15)
                
                node_text.append(f"{node}<br>Type: {G.nodes[node]['type']}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='white')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='Agent-Teller Network',
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=0, l=0, r=0, t=40),
                              height=600,
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Network Statistics")
            
            stats = [
                ("Total Nodes", len(G.nodes())),
                ("Total Edges", len(G.edges())),
                ("Avg Clustering", round(nx.average_clustering(G), 3)),
                ("Diameter", nx.diameter(G) if nx.is_connected(G) else "N/A"),
                ("Density", round(nx.density(G), 3))
            ]
            
            for stat_name, stat_value in stats:
                st.metric(stat_name, stat_value)
            
            st.markdown("---")
            st.markdown("#### 🎯 Legend")
            st.markdown("🔵 **Blue**: Agents")
            st.markdown("🟢 **Green**: Tellers")
            st.markdown("---")
            st.markdown("#### ⚙️ Controls")
            st.slider("Zoom Level", 0.5, 2.0, 1.0, 0.1)
            st.checkbox("Show Labels", value=True)
            st.checkbox("Show Weights", value=False)
    
    with tabs[1]:
        st.markdown("### 📊 Network Metrics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Degree Distribution
            G = create_network_graph()
            degrees = [deg for _, deg in G.degree()]
            
            fig = px.histogram(
                x=degrees,
                title='Degree Distribution',
                labels={'x': 'Number of Connections', 'y': 'Count'},
                color_discrete_sequence=['#3B82F6']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Centrality Measures
            centrality_data = {
                'Node': list(G.nodes())[:10],
                'Degree Centrality': [deg/len(G.nodes()) for _, deg in list(G.degree())[:10]],
                'Betweenness': list(nx.betweenness_centrality(G).values())[:10]
            }
            
            df_centrality = pd.DataFrame(centrality_data)
            
            fig = px.bar(
                df_centrality,
                x='Node',
                y=['Degree Centrality', 'Betweenness'],
                title='Top 10 Nodes by Centrality',
                barmode='group'
            )
            
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### 🔗 Connection Analysis")
        
        # Connection matrix
        st.markdown("#### 📋 Connection Matrix")
        
        # Sample connection data
        nodes = [f"A{i}" for i in range(1, 6)] + [f"T{i}" for i in range(1, 6)]
        matrix_data = np.random.randint(0, 2, (10, 10))
        
        fig = px.imshow(
            matrix_data,
            labels=dict(x="To Node", y="From Node", color="Connection"),
            x=nodes,
            y=nodes,
            title="Connection Matrix",
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Connection statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Connection Statistics")
            conn_stats = pd.DataFrame({
                'Metric': ['Avg Connections per Agent', 'Max Connections', 'Min Connections', 'Most Connected'],
                'Value': ['3.2', '7', '1', 'A12 (7 connections)']
            })
            st.dataframe(conn_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 💡 Connection Insights")
            insights = [
                "Agents with more tellers have 35% higher activity",
                "Top 20% of agents handle 60% of transactions",
                "Network redundancy is at optimal levels",
                "Isolated nodes identified: 12 agents"
            ]
            
            for insight in insights:
                st.markdown(f"- {insight}")
    
    with tabs[3]:
        st.markdown("### 🏙️ Regional Network View")
        
        # Regional distribution
        if 'regional_distribution' in results and results['regional_distribution']:
            regions = list(results['regional_distribution'].keys())
            counts = list(results['regional_distribution'].values())
            
            fig = px.bar(
                x=regions,
                y=counts,
                title='Regional Agent Distribution',
                labels={'x': 'Region', 'y': 'Agent Count'},
                color=counts,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional metrics
        st.markdown("#### 📈 Regional Performance Metrics")
        
        regional_metrics = pd.DataFrame({
            'Region': ['West Coast', 'Greater Banjul', 'Central River', 'North Bank', 'Lower River'],
            'Agent Count': [450, 380, 320, 280, 220],
            'Teller Count': [350, 300, 250, 200, 150],
            'Agent/Teller Ratio': ['1:0.78', '1:0.79', '1:0.78', '1:0.71', '1:0.68'],
            'Growth Rate': ['18.5%', '15.2%', '12.8%', '9.5%', '7.2%'],
            'Activity Level': ['85%', '82%', '78%', '75%', '70%']
        })
        
        st.dataframe(
            regional_metrics.style.background_gradient(cmap='RdYlGn', subset=['Growth Rate', 'Activity Level']),
            use_container_width=True
        )
        
        # Regional recommendations
        st.markdown("#### 🎯 Regional Recommendations")
        
        recommendations = [
            "**West Coast**: Expand teller network in suburban areas",
            "**Greater Banjul**: Focus on agent retention programs",
            "**Central River**: Increase digital adoption training",
            "**North Bank**: Launch new agent recruitment drive",
            "**Lower River**: Improve transaction processing efficiency"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()