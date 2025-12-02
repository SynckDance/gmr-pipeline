"""
GMR Pipeline - Part B: 3D Humanoid Visualization
=================================================
This module generates interactive 3D stick figure animations from mocap data.

Output: A self-contained HTML file using Three.js that can be hosted anywhere.
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Skeleton graph definition: which joints connect to which
SKELETON_CONNECTIONS = [
    # Spine
    ('center_of_gravity', 'spine'),
    ('spine', 'spine1'),
    ('spine1', 'spine2'),
    ('spine2', 'spine3'),
    ('spine3', 'spine4'),
    ('spine4', 'neck'),
    ('neck', 'head'),
    ('head', 'head_end'),
    
    # Left arm
    ('spine3', 'shoulder_L'),
    ('shoulder_L', 'elbow_L'),
    ('elbow_L', 'wrist_L'),
    ('wrist_L', 'hand_L'),
    
    # Right arm
    ('spine3', 'shoulder_R'),
    ('shoulder_R', 'elbow_R'),
    ('elbow_R', 'wrist_R'),
    ('wrist_R', 'hand_R'),
    
    # Left leg
    ('center_of_gravity', 'hip_L'),
    ('hip_L', 'knee_L'),
    ('knee_L', 'ankle_L'),
    ('ankle_L', 'toe_L'),
    
    # Right leg
    ('center_of_gravity', 'hip_R'),
    ('hip_R', 'knee_R'),
    ('knee_R', 'ankle_R'),
    ('ankle_R', 'toe_R'),
]

# Joint colors for visualization
JOINT_COLORS = {
    'head': '#FF6B6B',      # Red
    'shoulder': '#4ECDC4',  # Teal
    'elbow': '#45B7D1',     # Light blue
    'wrist': '#96CEB4',     # Green
    'hand': '#FFEAA7',      # Yellow
    'hip': '#DDA0DD',       # Plum
    'knee': '#98D8C8',      # Mint
    'ankle': '#F7DC6F',     # Gold
    'toe': '#BB8FCE',       # Purple
    'spine': '#85C1E9',     # Sky blue
    'default': '#FFFFFF'    # White
}


def get_joint_color(joint_name: str) -> str:
    """Get color for a joint based on its name."""
    for key in JOINT_COLORS:
        if key in joint_name.lower():
            return JOINT_COLORS[key]
    return JOINT_COLORS['default']


def df_to_frame_data(df: pd.DataFrame) -> List[Dict]:
    """
    Convert DataFrame to frame-by-frame joint position data for JavaScript.
    
    Args:
        df: Standardized DataFrame from data_cleaning module
        
    Returns:
        List of frame dictionaries with joint positions
    """
    frames = []
    
    # Get all unique joint names (excluding coordinate suffixes)
    joint_names = set()
    for col in df.columns:
        if col != 'frame' and '_' in col:
            joint_name = '_'.join(col.split('_')[:-1])
            joint_names.add(joint_name)
    
    for idx in range(len(df)):
        frame_data = {}
        for joint in joint_names:
            x_col = f"{joint}_x"
            y_col = f"{joint}_y"
            z_col = f"{joint}_z"
            
            if x_col in df.columns and y_col in df.columns and z_col in df.columns:
                frame_data[joint] = {
                    'x': float(df.iloc[idx][x_col]),
                    'y': float(df.iloc[idx][y_col]),
                    'z': float(df.iloc[idx][z_col])
                }
        frames.append(frame_data)
    
    return frames


def generate_threejs_html(df: pd.DataFrame, title: str = "Dance Visualization", fps: float = 30.0) -> str:
    """
    Generate a complete HTML file with Three.js visualization.
    
    Args:
        df: Standardized DataFrame with joint positions
        title: Title for the visualization
        fps: Frames per second for playback
        
    Returns:
        Complete HTML string
    """
    # Convert data to JSON
    frame_data = df_to_frame_data(df)
    
    # Get valid connections (only those with data)
    available_joints = set(frame_data[0].keys()) if frame_data else set()
    valid_connections = [
        conn for conn in SKELETON_CONNECTIONS
        if conn[0] in available_joints and conn[1] in available_joints
    ]
    
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            overflow: hidden;
        }}
        #container {{ width: 100vw; height: 100vh; }}
        #controls {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 15px;
            align-items: center;
            background: rgba(0,0,0,0.7);
            padding: 15px 25px;
            border-radius: 50px;
            backdrop-filter: blur(10px);
        }}
        button {{
            background: #4ECDC4;
            border: none;
            color: #1a1a2e;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        button:hover {{ background: #45B7D1; transform: scale(1.05); }}
        button.active {{ background: #FF6B6B; }}
        #frame-info {{
            font-size: 14px;
            min-width: 120px;
            text-align: center;
        }}
        #speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        input[type="range"] {{
            width: 100px;
            accent-color: #4ECDC4;
        }}
        #title {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 24px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        #instructions {{
            position: fixed;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div id="title">{title}</div>
    <div id="instructions">Drag to rotate | Scroll to zoom | Right-click drag to pan</div>
    <div id="container"></div>
    <div id="controls">
        <button id="playPause">▶ Play</button>
        <button id="reset">⟲ Reset</button>
        <span id="frame-info">Frame: 0 / {len(frame_data)}</span>
        <div id="speed-control">
            <span>Speed:</span>
            <input type="range" id="speed" min="0.1" max="3" step="0.1" value="1">
            <span id="speed-value">1.0x</span>
        </div>
        <button id="loop" class="active">🔁 Loop</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Motion capture data
        const frameData = {json.dumps(frame_data)};
        const connections = {json.dumps(valid_connections)};
        const fps = {fps};
        
        // Three.js setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
        camera.position.set(0, 1000, 2000);
        camera.lookAt(0, 500, 0);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(500, 1000, 500);
        scene.add(directionalLight);
        
        // Grid helper
        const gridHelper = new THREE.GridHelper(2000, 20, 0x444444, 0x333333);
        scene.add(gridHelper);
        
        // Joint spheres and bone lines
        const joints = {{}};
        const bones = [];
        
        // Create joint spheres
        const jointGeometry = new THREE.SphereGeometry(15, 16, 16);
        if (frameData.length > 0) {{
            for (const jointName of Object.keys(frameData[0])) {{
                const color = getJointColor(jointName);
                const material = new THREE.MeshPhongMaterial({{ color: color, emissive: color, emissiveIntensity: 0.3 }});
                const sphere = new THREE.Mesh(jointGeometry, material);
                joints[jointName] = sphere;
                scene.add(sphere);
            }}
        }}
        
        // Create bone lines
        const boneMaterial = new THREE.LineBasicMaterial({{ color: 0x4ECDC4, linewidth: 2 }});
        for (const [j1, j2] of connections) {{
            if (joints[j1] && joints[j2]) {{
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(6);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const line = new THREE.Line(geometry, boneMaterial);
                bones.push({{ line, j1, j2 }});
                scene.add(line);
            }}
        }}
        
        function getJointColor(name) {{
            const colors = {{
                head: 0xFF6B6B, shoulder: 0x4ECDC4, elbow: 0x45B7D1, wrist: 0x96CEB4,
                hand: 0xFFEAA7, hip: 0xDDA0DD, knee: 0x98D8C8, ankle: 0xF7DC6F,
                toe: 0xBB8FCE, spine: 0x85C1E9, center: 0xFFFFFF
            }};
            for (const [key, color] of Object.entries(colors)) {{
                if (name.toLowerCase().includes(key)) return color;
            }}
            return 0xFFFFFF;
        }}
        
        // Orbit controls (simple implementation)
        let isDragging = false;
        let previousMousePosition = {{ x: 0, y: 0 }};
        let theta = 0, phi = Math.PI / 4;
        let radius = 2500;
        
        container.addEventListener('mousedown', (e) => {{
            isDragging = true;
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
        }});
        
        container.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            theta -= deltaX * 0.005;
            phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi + deltaY * 0.005));
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
            updateCameraPosition();
        }});
        
        container.addEventListener('mouseup', () => isDragging = false);
        container.addEventListener('mouseleave', () => isDragging = false);
        
        container.addEventListener('wheel', (e) => {{
            radius = Math.max(500, Math.min(5000, radius + e.deltaY));
            updateCameraPosition();
        }});
        
        function updateCameraPosition() {{
            camera.position.x = radius * Math.sin(phi) * Math.sin(theta);
            camera.position.y = radius * Math.cos(phi);
            camera.position.z = radius * Math.sin(phi) * Math.cos(theta);
            camera.lookAt(0, 500, 0);
        }}
        
        // Animation state
        let currentFrame = 0;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        let isLooping = true;
        let lastTime = 0;
        
        const playPauseBtn = document.getElementById('playPause');
        const resetBtn = document.getElementById('reset');
        const frameInfo = document.getElementById('frame-info');
        const speedSlider = document.getElementById('speed');
        const speedValue = document.getElementById('speed-value');
        const loopBtn = document.getElementById('loop');
        
        playPauseBtn.onclick = () => {{
            isPlaying = !isPlaying;
            playPauseBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            playPauseBtn.classList.toggle('active', isPlaying);
        }};
        
        resetBtn.onclick = () => {{
            currentFrame = 0;
            updateFrame();
        }};
        
        speedSlider.oninput = () => {{
            playbackSpeed = parseFloat(speedSlider.value);
            speedValue.textContent = playbackSpeed.toFixed(1) + 'x';
        }};
        
        loopBtn.onclick = () => {{
            isLooping = !isLooping;
            loopBtn.classList.toggle('active', isLooping);
        }};
        
        function updateFrame() {{
            if (frameData.length === 0) return;
            
            const frame = frameData[Math.floor(currentFrame)];
            
            // Update joint positions
            for (const [jointName, pos] of Object.entries(frame)) {{
                if (joints[jointName]) {{
                    joints[jointName].position.set(pos.x, pos.y, pos.z);
                }}
            }}
            
            // Update bone lines
            for (const bone of bones) {{
                const p1 = joints[bone.j1]?.position;
                const p2 = joints[bone.j2]?.position;
                if (p1 && p2) {{
                    const positions = bone.line.geometry.attributes.position.array;
                    positions[0] = p1.x; positions[1] = p1.y; positions[2] = p1.z;
                    positions[3] = p2.x; positions[4] = p2.y; positions[5] = p2.z;
                    bone.line.geometry.attributes.position.needsUpdate = true;
                }}
            }}
            
            frameInfo.textContent = `Frame: ${{Math.floor(currentFrame) + 1}} / ${{frameData.length}}`;
        }}
        
        function animate(time) {{
            requestAnimationFrame(animate);
            
            if (isPlaying) {{
                const deltaTime = (time - lastTime) / 1000;
                currentFrame += deltaTime * fps * playbackSpeed;
                
                if (currentFrame >= frameData.length) {{
                    if (isLooping) {{
                        currentFrame = 0;
                    }} else {{
                        currentFrame = frameData.length - 1;
                        isPlaying = false;
                        playPauseBtn.textContent = '▶ Play';
                        playPauseBtn.classList.remove('active');
                    }}
                }}
                
                updateFrame();
            }}
            
            lastTime = time;
            renderer.render(scene, camera);
        }}
        
        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Initialize
        updateCameraPosition();
        updateFrame();
        animate(0);
    </script>
</body>
</html>'''
    
    return html_template


def create_visualization(df: pd.DataFrame, output_path: str, title: str = "Dance Visualization", fps: float = 30.0):
    """
    Create and save an HTML visualization file.
    
    Args:
        df: Standardized DataFrame with joint positions
        output_path: Path to save the HTML file
        title: Title for the visualization
        fps: Frames per second for playback
    """
    html = generate_threejs_html(df, title, fps)
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Visualization saved to {output_path}")
    print(f"Open this file in a web browser to view the animation.")


if __name__ == "__main__":
    # Test with the data cleaning module
    from data_cleaning import process_dance_csv
    
    # Process a dance
    df, summary = process_dance_csv("/mnt/user-data/uploads/Bata_dance_Sinclair.csv")
    
    # Create visualization
    create_visualization(
        df, 
        "bata_dance_viz.html",
        title="Bata Dance - Sinclair",
        fps=summary['target_fps']
    )
