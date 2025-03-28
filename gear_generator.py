import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import sys
import os
import argparse

# Add the submodule path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/gear-profile-generator'))
import gear

class InvoluteGear:
    def __init__(self, module, teeth_number, pressure_angle=20.0, clearance=0.25, backlash=0.0, profile_shift=0):
        """
        Involute gear generation class
        
        Parameters:
        -----------
        module : float
            Module (unit that determines tooth size)
        teeth_number : int
            Number of teeth
        pressure_angle : float
            Pressure angle (degrees)
        clearance : float
            Clearance coefficient
        backlash : float
            Backlash
        profile_shift : float
            Profile shift coefficient
        """
        self.module = module
        self.teeth_number = teeth_number
        self.pressure_angle = pressure_angle
        self.clearance = clearance
        self.backlash = backlash
        self.profile_shift = profile_shift
        
        # Calculate tooth width from module
        self.tooth_width = np.pi * module
        
        # Generate gear profile
        self.gear_poly, self.pitch_radius = gear.generate(
            teeth_count=teeth_number,
            tooth_width=self.tooth_width,
            pressure_angle=gear.deg2rad(pressure_angle),
            backlash=backlash,
            frame_count=32  # Smoothness of the curve
        )
    
    def points(self):
        """
        Get points of the gear profile
        
        Returns:
        --------
        numpy.ndarray
            Array of gear profile points
        """
        # Get exterior points from the Polygon object in gear.py
        exterior_coords = np.array(self.gear_poly.exterior.coords)
        return exterior_coords

def create_gear_patch(gear, hole_radius=0.2, facecolor='lightgray'):
    """
    Create a Matplotlib patch from a gear profile
    
    Parameters:
    -----------
    gear : InvoluteGear
        Gear profile object
    hole_radius : float
        Radius of the center hole
    facecolor : str
        Color of the gear
        
    Returns:
    --------
    matplotlib.patches.PathPatch
        Patch representing the gear
    """
    # Get points of the gear profile
    points = gear.points()
    
    # Lists for vertices and codes
    vertices = []
    codes = []
    
    # Add the outer perimeter of the gear
    for i, point in enumerate(points):
        if i == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
        vertices.append((point[0], point[1]))
    
    # Close the path
    codes.append(Path.CLOSEPOLY)
    vertices.append(vertices[0])
    
    # Add center hole
    theta = np.linspace(0, 2*np.pi, 50)
    first_hole_point = True
    for t in theta:
        x = hole_radius * np.cos(t)
        y = hole_radius * np.sin(t)
        if first_hole_point:
            codes.append(Path.MOVETO)
            first_hole_point = False
        else:
            codes.append(Path.LINETO)
        vertices.append((x, y))
    
    # Close the hole
    codes.append(Path.CLOSEPOLY)
    vertices.append((hole_radius, 0))
    
    # Create path
    path = Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor=facecolor, edgecolor='black', lw=1.5)
    
    return patch, gear.pitch_radius

def export_gears_only(module, teeth1, teeth2, pressure_angle, clearance, backlash, filename):
    """
    Export only the gears to an SVG file without any additional elements
    
    Parameters:
    -----------
    module : float
        Module (tooth size unit)
    teeth1 : int
        Number of teeth for the large gear
    teeth2 : int
        Number of teeth for the small gear
    pressure_angle : float
        Pressure angle in degrees
    clearance : float
        Clearance coefficient
    backlash : float
        Backlash
    filename : str
        Output filename
    """
    # Create a figure with transparent background
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='none')
    
    # Large gear
    gear1_obj = InvoluteGear(
        module=module,
        teeth_number=teeth1,
        pressure_angle=pressure_angle,
        clearance=clearance,
        backlash=backlash,
        profile_shift=0
    )
    gear1, pitch_radius1 = create_gear_patch(gear1_obj, hole_radius=3.0)
    ax.add_patch(gear1)

    # Small gear
    gear2_obj = InvoluteGear(
        module=module,
        teeth_number=teeth2,
        pressure_angle=pressure_angle,
        clearance=clearance,
        backlash=backlash,
        profile_shift=0
    )
    center_distance = gear1_obj.pitch_radius + gear2_obj.pitch_radius

    # Calculate rotation angle for the small gear (for meshing)
    rotation_angle = np.pi / teeth2  # Half tooth rotation

    # Generate and position the small gear
    gear2, pitch_radius2 = create_gear_patch(gear2_obj, hole_radius=1.5, facecolor='darkgray')
    transform = patches.Affine2D().rotate(rotation_angle).translate(center_distance, 0)
    ax.add_patch(patches.PathPatch(gear2.get_path(), facecolor='darkgray', edgecolor='black', lw=1.5, transform=transform + ax.transData))

    # Set equal axes
    margin = 1.2
    ax.set_xlim(-pitch_radius1 * margin, center_distance + pitch_radius2 * margin)
    ax.set_ylim(-pitch_radius1 * margin, pitch_radius1 * margin)
    ax.set_aspect('equal')
    
    # Remove all axes, grid, and other elements
    ax.axis('off')
    
    # Save the figure with transparent background
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    print(f"Gears exported to {filename}")
    plt.close(fig)

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Generate and visualize involute gears')
    parser.add_argument('--module', type=float, default=1.0, help='Module (tooth size unit)')
    parser.add_argument('--teeth1', type=int, default=30, help='Number of teeth for the large gear')
    parser.add_argument('--teeth2', type=int, default=15, help='Number of teeth for the small gear')
    parser.add_argument('--pressure-angle', type=float, default=20.0, help='Pressure angle in degrees')
    parser.add_argument('--backlash', type=float, default=0.0, help='Backlash')
    parser.add_argument('--clearance', type=float, default=0.25, help='Clearance coefficient')
    parser.add_argument('--save', type=str, help='Save figure to file (SVG, PNG, PDF, etc.)')
    parser.add_argument('--gears-only', type=bool, default=True, help='Export only the gears to a file')
    parser.add_argument('--show', type=bool, default=True, help='Show the figure')
    args = parser.parse_args()

    # If gears-only mode is requested and a filename is provided
    if args.gears_only and args.save:
        export_gears_only(
            args.module, 
            args.teeth1, 
            args.teeth2, 
            args.pressure_angle, 
            args.clearance, 
            args.backlash, 
            args.save
        )
        return

    # Create figure for normal display
    fig, ax = plt.subplots(figsize=(12, 10))

    # Gear parameters
    module = args.module
    pressure_angle = args.pressure_angle

    # Large gear
    num_teeth1 = args.teeth1
    # Create InvoluteGear object
    gear1_obj = InvoluteGear(
        module=module,
        teeth_number=num_teeth1,
        pressure_angle=pressure_angle,
        clearance=args.clearance,
        backlash=args.backlash,
        profile_shift=0
    )
    gear1, pitch_radius1 = create_gear_patch(gear1_obj, hole_radius=3.0)
    ax.add_patch(gear1)

    # Small gear
    num_teeth2 = args.teeth2
    # Create InvoluteGear object
    gear2_obj = InvoluteGear(
        module=module,
        teeth_number=num_teeth2,
        pressure_angle=pressure_angle,
        clearance=args.clearance,
        backlash=args.backlash,
        profile_shift=0
    )
    center_distance = gear1_obj.pitch_radius + gear2_obj.pitch_radius  # Theoretical center distance

    # Calculate rotation angle for the small gear (for meshing)
    rotation_angle = np.pi / num_teeth2  # Half tooth rotation

    # Generate and position the small gear
    gear2, pitch_radius2 = create_gear_patch(gear2_obj, hole_radius=1.5, facecolor='darkgray')
    transform = patches.Affine2D().rotate(rotation_angle).translate(center_distance, 0)
    ax.add_patch(patches.PathPatch(gear2.get_path(), facecolor='darkgray', edgecolor='black', lw=1.5, transform=transform + ax.transData))

    # Display pitch circles (for reference)
    pitch_circle1 = plt.Circle((0, 0), pitch_radius1, fill=False, color='red', linestyle='--', alpha=0.7)
    pitch_circle2 = plt.Circle((center_distance, 0), pitch_radius2, fill=False, color='red', linestyle='--', alpha=0.7)
    ax.add_patch(pitch_circle1)
    ax.add_patch(pitch_circle2)

    # Set equal axes
    margin = 1.2
    ax.set_xlim(-pitch_radius1 * margin, center_distance + pitch_radius2 * margin)
    ax.set_ylim(-pitch_radius1 * margin, pitch_radius1 * margin)
    ax.set_aspect('equal')

    # Display grid and axes
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=center_distance, color='k', linestyle='-', alpha=0.3)

    # Add title
    plt.title('Involute Gear Meshing Design', fontsize=16)

    # Explanatory text
    ax.text(0, -pitch_radius1 * 1.1, f'Large Gear: {num_teeth1} teeth', ha='center', fontsize=12)
    ax.text(center_distance, -pitch_radius2 * 1.1, f'Small Gear: {num_teeth2} teeth', ha='center', fontsize=12)
    ax.text(center_distance/2, pitch_radius1 * 1.1, f'Pressure Angle: {pressure_angle}°', ha='center', fontsize=12)

    plt.tight_layout()
    
    # Save figure if requested
    if args.save and not args.gears_only:
        plt.savefig(args.save, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {args.save}")
    
    # Show the figure if requested
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
