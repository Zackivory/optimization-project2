from pprint import pprint
import pandas as pd
import networkx as nx
from haversine import haversine, Unit
from gurobipy import *
import gurobipy as gp
import numpy as np
import random

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Load the data
data = pd.read_csv('airports.csv')
# Add one to the first column value
data['Unnamed: 0'] = data['Unnamed: 0'] + 1
def optimizer_case_1(data,accuracy=0.08):
    """
    This function is used to optimize using problem 2 case 1 setting which is two planes - one starting/landing at JFK, the other one starting/landing at LAX
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes
    for index, row in data.iterrows():
        G.add_node(row['Unnamed: 0'], country=row['country'], airport_name=row['airport_name'], airport_code=row['airport_code'],
                   latitude_deg=row['latitude_deg'], longitude_deg=row['longitude_deg'])

    # Add edges with haversine distance as weight
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i != j:
                coord1 = (row1['latitude_deg'], row1['longitude_deg'])
                coord2 = (row2['latitude_deg'], row2['longitude_deg'])
                distance = haversine(coord1, coord2, unit=Unit.MILES)
                G.add_edge(row1['Unnamed: 0'], row2['Unnamed: 0'], weight=distance)


    # Create a new model
    mtz = gp.Model("TSP_MTZ")
    mtz.Params.MIPGap = accuracy  # Set the target accuracy to 8% of the optimal

    # Create variables
    # also satisfy constraint 4 and 5
    vars = mtz.addVars([(i, j, k) for i in G.nodes for j in G.nodes for k in [1, 2] if i != j],
                         vtype=GRB.BINARY, name="e")

    # Subtour elimination decision variables
    u = mtz.addVars([(i, k) for i in G.nodes for k in [1, 2]], vtype=GRB.CONTINUOUS, name="u", lb=1, ub=len(G.nodes))

    # Objective: minimize the total distance traveled
    mtz.setObjective(gp.quicksum(vars[i, j, k] * G[i][j]['weight']
                                   for i in G.nodes for j in G.nodes for k in [1, 2] if i != j), GRB.MINIMIZE)

    # Constraints
    # 1.and 2.
    # Each node is entered and left exactly once
    # for i in G.nodes:
    #     for k in [1, 2]:
    #         mtz.addConstr(gp.quicksum(vars[i, j, k] for j in G.nodes if i != j) == 1)
    #         mtz.addConstr(gp.quicksum(vars[j, i, k] for j in G.nodes if i != j) == 1)
    for i in G.nodes:
            mtz.addConstr(gp.quicksum(vars[i, j, k] for k in [1,2] for j in G.nodes if i != j) == 1)
            mtz.addConstr(gp.quicksum(vars[j, i, k] for k in [1,2] for j in G.nodes if i != j) == 1)

    #3. Plane-Specific Flow Conservation Constraints
    for i in G.nodes:
        if i not in [1, 2]:
            for k in [1, 2]:
                mtz.addConstr(quicksum(vars[i, j, k] for j in G.nodes if i != j) == quicksum(vars[j, i, k] for j in G.nodes if i != j))

    # 4. Subtour elimination constraints (MTZ)
    n = len(G.nodes)
    for k in [1, 2]:
        for i in list(G.nodes)[2:]:  # Skip the first node
            for j in list(G.nodes)[2:]:
                if i != j:
                    mtz.addConstr(u[i,k] + 1 <= u[j,k] + (n - 1) * (1 - vars[i, j, k]))
    for k in [1, 2]:
        mtz.addConstr(u[k, k] == 1)  # Fix the position of the first node for all k in K

    # 7.order constraint
    for k in [1, 2]:
        for i in list(G.nodes)[2:]:
            mtz.addConstr(u[i, k] >= 2)
            mtz.addConstr(u[i, k] <= len(G.nodes))

    # 8. Start and End at Their Respective Depots
    # Departure from Depot
    for k in [1, 2]:
        # mtz.addConstr(gp.quicksum(vars[k, j, k] for j in G.nodes if k != j) == 1)
        mtz.addConstr(gp.quicksum(vars[k, j, k] for j in G.nodes if (j != 1 and j != 2)) == 1)

    # Return to Depot
    for k in [1, 2]:
        mtz.addConstr(gp.quicksum(vars[i, k, k] for i in G.nodes if (i != 1 and i !=2)) == 1)


    # Solve the model
    mtz.optimize()

    # Output the solution
    if mtz.status == GRB.OPTIMAL:
        edges = [(i, j, k) for i in G.nodes for j in G.nodes for k in [1, 2] if i != j and vars[i, j, k].X > 0.5]
        return edges,G

    else:
        print("No optimal solution found")
        return None,None

def optimizer_case2_JFK_based(data,accuracy=0.08):
    # Create a new model
    mtz1 = gp.Model("TSP_JFK")
    mtz1.Params.MIPGap = accuracy
    # Construct a new graph
    G_JFK = nx.Graph()
    # Create a graph
    G = nx.Graph()

    # Add nodes
    for index, row in data.iterrows():
        G.add_node(row['Unnamed: 0'], country=row['country'], airport_name=row['airport_name'],
                   airport_code=row['airport_code'],
                   latitude_deg=row['latitude_deg'], longitude_deg=row['longitude_deg'])

    # Add edges with haversine distance as weight
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i != j:
                coord1 = (row1['latitude_deg'], row1['longitude_deg'])
                coord2 = (row2['latitude_deg'], row2['longitude_deg'])
                distance = haversine(coord1, coord2, unit=Unit.MILES)
                G.add_edge(row1['Unnamed: 0'], row2['Unnamed: 0'], weight=distance)
    for node, attrs in G.nodes(data=True):
        if node != 2:
            G_JFK.add_node(node, **attrs)

    # Add edges with haversine distance as weight
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i != j and i != 1 and j != 1:
                coord1 = (row1['latitude_deg'], row1['longitude_deg'])
                coord2 = (row2['latitude_deg'], row2['longitude_deg'])
                distance = haversine(coord1, coord2, unit=Unit.MILES)
                G_JFK.add_edge(row1['Unnamed: 0'], row2['Unnamed: 0'], weight=distance)
    # Create variables
    # also satisfy constraint 4 and 5
    vars = mtz1.addVars([(i, j) for i in G_JFK.nodes for j in G_JFK.nodes if i != j],
                        vtype=GRB.BINARY, name="e")

    # Subtour elimination decision variables
    u = mtz1.addVars(G_JFK.nodes, vtype=GRB.CONTINUOUS, name="u", lb=1, ub=len(G_JFK.nodes))

    # Objective: minimize the total distance traveled
    mtz1.setObjective(gp.quicksum(vars[i, j] * G_JFK[i][j]['weight']
                                  for i in G_JFK.nodes for j in G_JFK.nodes if i != j), GRB.MINIMIZE)

    # Constraints
    # 1.and 2.
    # Each node is entered and left exactly once
    for i in G_JFK.nodes:
        mtz1.addConstr(gp.quicksum(vars[i, j] for j in G_JFK.nodes if i != j) == 1)
        mtz1.addConstr(gp.quicksum(vars[j, i] for j in G_JFK.nodes if i != j) == 1)

    # 3. Subtour elimination constraints (MTZ)
    n = len(G_JFK.nodes)
    for i in list(G_JFK.nodes)[1:]:  # Skip the first node
        for j in list(G_JFK.nodes)[1:]:
            if i != j:
                mtz1.addConstr(u[i] - u[j] + (n - 1) * vars[i, j] <= n - 2)
    mtz1.addConstr(u[1] == 1)  # Fix the position of the first node

    # 6.order constraint
    for i in list(G_JFK.nodes)[1:]:
        mtz1.addConstr(u[i] >= 2)
        mtz1.addConstr(u[i] <= len(G_JFK.nodes))

    # Solve the model
    mtz1.optimize()

    # Output the solution
    if mtz1.status == GRB.OPTIMAL:
        edges = [(i, j) for i in G_JFK.nodes for j in G_JFK.nodes if i != j and vars[i, j].X > 0.5]
        return edges, G_JFK
    else:
        print("No optimal solution found")
        return None,None

def optimizer_case2_LAX_based(data,accuracy=0.08):
    # Construct a new graph
    G_LAX = nx.Graph()
    # Create a graph
    G = nx.Graph()

    # Add nodes
    for index, row in data.iterrows():
        G.add_node(row['Unnamed: 0'], country=row['country'], airport_name=row['airport_name'],
                   airport_code=row['airport_code'],
                   latitude_deg=row['latitude_deg'], longitude_deg=row['longitude_deg'])

    # Add edges with haversine distance as weight
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i != j:
                coord1 = (row1['latitude_deg'], row1['longitude_deg'])
                coord2 = (row2['latitude_deg'], row2['longitude_deg'])
                distance = haversine(coord1, coord2, unit=Unit.MILES)
                G.add_edge(row1['Unnamed: 0'], row2['Unnamed: 0'], weight=distance)
    for node, attrs in G.nodes(data=True):
        if node != 1:
            G_LAX.add_node(node, **attrs)

    # Add edges with haversine distance as weight
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i != j and i != 0 and j != 0:
                coord1 = (row1['latitude_deg'], row1['longitude_deg'])
                coord2 = (row2['latitude_deg'], row2['longitude_deg'])
                distance = haversine(coord1, coord2, unit=Unit.MILES)
                G_LAX.add_edge(row1['Unnamed: 0'], row2['Unnamed: 0'], weight=distance)
    # Create a new model
    mtz2 = gp.Model("TSP_LAX")
    mtz2.Params.MIPGap = accuracy

    # Create variables
    # also satisfy constraint 4 and 5
    vars = mtz2.addVars([(i, j) for i in G_LAX.nodes for j in G_LAX.nodes if i != j],
                        vtype=GRB.BINARY, name="e")

    # Subtour elimination decision variables
    u = mtz2.addVars(G_LAX.nodes, vtype=GRB.CONTINUOUS, name="u", lb=1, ub=len(G_LAX.nodes))

    # Objective: minimize the total distance traveled
    mtz2.setObjective(gp.quicksum(vars[i, j] * G_LAX[i][j]['weight']
                                  for i in G_LAX.nodes for j in G_LAX.nodes if i != j), GRB.MINIMIZE)

    # Constraints
    # 1.and 2.
    # Each node is entered and left exactly once
    for i in G_LAX.nodes:
        mtz2.addConstr(gp.quicksum(vars[i, j] for j in G_LAX.nodes if i != j) == 1)
        mtz2.addConstr(gp.quicksum(vars[j, i] for j in G_LAX.nodes if i != j) == 1)

    # 3. Subtour elimination constraints (MTZ)
    n = len(G_LAX.nodes)
    for i in list(G_LAX.nodes)[1:]:  # Skip the first node
        for j in list(G_LAX.nodes)[1:]:
            if i != j:
                mtz2.addConstr(u[i] - u[j] + (n - 1) * vars[i, j] <= n - 2)
    mtz2.addConstr(u[2] == 1)  # Fix the position of the first node

    # 6.order constraint
    for i in list(G_LAX.nodes)[1:]:
        mtz2.addConstr(u[i] >= 2)
        mtz2.addConstr(u[i] <= len(G_LAX.nodes))

    # Solve the model
    mtz2.optimize()

    # Output the solution
    if mtz2.status == GRB.OPTIMAL:
        edges = [(i, j) for i in G_LAX.nodes for j in G_LAX.nodes if i != j and vars[i, j].X > 0.5]
        return edges, G_LAX

    else:
        print("No optimal solution found")
        return None,None

def plot(edges,G):
    # Define the optimal tour provided by gurobi
    optimal_tour_k1 = [(i, j) for i, j, k in edges if k == 1]  # Extract node pairs for k=1
    optimal_tour_k2 = [(i, j) for i, j, k in edges if k == 2]  # Extract node pairs for k=2

    # Draw the network with the optimal path highlighted
    G_optimal_k1 = nx.DiGraph()
    G_optimal_k1.add_edges_from(optimal_tour_k1)
    G_optimal_k2 = nx.DiGraph()
    G_optimal_k2.add_edges_from(optimal_tour_k2)

    # Define the positions of the nodes based on latitude and longitude
    pos = {node: (G.nodes[node]['longitude_deg'], G.nodes[node]['latitude_deg']) for node in G.nodes}

    # Draw the full graph with all nodes
    nx.draw(G, pos, with_labels=True, node_color='orange', node_size=5, edge_color='gray', arrows=True)

    # Overlay the optimal tour for k=1 with a different edge color
    nx.draw_networkx_edges(G_optimal_k1, pos, edge_color='teal', width=2, arrows=True, arrowstyle='-|>', arrowsize=20)

    # Overlay the optimal tour for k=2 with another different edge color
    nx.draw_networkx_edges(G_optimal_k2, pos, edge_color='purple', width=2, arrows=True, arrowstyle='-|>', arrowsize=20)

    plt.title('Optimal Tour using MTZ within 8% of optimality ')
    plt.show()

def plot_on_map(edges,data,title):
    # Define the optimal tour provided by gurobi
    optimal_tour_k1 = [(i, j) for i, j, k in edges if k == 1]  # Extract node pairs for k=1
    optimal_tour_k2 = [(i, j) for i, j, k in edges if k == 2]  # Extract node pairs for k=2

    # 提取节点数据
    node_data = {row['Unnamed: 0']: (row['latitude_deg'], row['longitude_deg'], row['airport_code'])
                 for _, row in data.iterrows()}

    # 设置地图投影
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)

    # 绘制节点
    for node, (lat, lon, name) in node_data.items():
        ax.plot(lon, lat, 'bo', markersize=5, transform=ccrs.PlateCarree())  # 绘制节点
        ax.text(lon + 1, lat, f"{name}", fontsize=10, transform=ccrs.PlateCarree())  # 标注节点信息

    # 绘制路径，按飞机区分颜色
    colors = {1: 'teal', 2: 'purple'}  # 为不同飞机分配颜色
    for i, j in optimal_tour_k1:
        lat1, lon1, _ = node_data[i]
        lat2, lon2, _ = node_data[j]
        ax.plot([lon1, lon2], [lat1, lat2], color=colors[1], linewidth=2, transform=ccrs.PlateCarree(),
                label=f"Plane {1}")
    for i, j in optimal_tour_k2:
        lat1, lon1, _ = node_data[i]
        lat2, lon2, _ = node_data[j]
        ax.plot([lon1, lon2], [lat1, lat2], color=colors[2], linewidth=2, transform=ccrs.PlateCarree(),
                label=f"Plane {2}")

    # 去除重复的图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    # 设置标题并显示地图
    plt.title(title, fontsize=16)
    plt.show()

def plot_on_map_one_plane(edges,data,title):
    # 提取节点数据
    node_data = {row['Unnamed: 0']: (row['latitude_deg'], row['longitude_deg'], row['airport_code'])
                 for _, row in data.iterrows()}

    # 设置地图投影
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)

    # 绘制节点
    for node, (lat, lon, name) in node_data.items():
        ax.plot(lon, lat, 'bo', markersize=5, transform=ccrs.PlateCarree())  # 绘制节点
        ax.text(lon + 1, lat, f"{name}", fontsize=10, transform=ccrs.PlateCarree())  # 标注节点信息

    # 绘制路径
    for i, j in edges:  # optimal_tour 来自求解结果
        lat1, lon1, _ = node_data[i]
        lat2, lon2, _ = node_data[j]
        ax.plot([lon1, lon2], [lat1, lat2], color='teal', linewidth=2, transform=ccrs.PlateCarree())  # 绘制路径

    # 设置标题和显示地图
    plt.title(title, fontsize=16)
    plt.show()