import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.path import Path
import sys
import os
import argparse
import math
from scipy.spatial.transform import Rotation as R

# Add the submodule path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/gear-profile-generator'))
import gear

# For 3DM export
try:
    import rhinoinside
    rhinoinside.load()
    import Rhino
    import Rhino.Geometry as rg
    RHINO_AVAILABLE = True
except ImportError:
    print("Warning: rhinoinside module not found. 3DM export will not be available.")
    RHINO_AVAILABLE = False

class BevelGear:
    def __init__(self, module, teeth_number, pressure_angle=20.0, cone_angle=45.0, 
                 face_width=10.0, clearance=0.25, backlash=0.0, profile_shift=0, frame_count=16):
        """
        Bevel gear generation class
        
        Parameters:
        -----------
        module : float
            Module (unit that determines tooth size)
        teeth_number : int
            Number of teeth
        pressure_angle : float
            Pressure angle (degrees)
        cone_angle : float
            Cone angle (degrees) - typically 45° for 90° mating gears
        face_width : float
            Width of the gear face along the cone
        clearance : float
            Clearance coefficient
        backlash : float
            Backlash
        profile_shift : float
            Profile shift coefficient
        frame_count : int
            Number of points to generate per curve (controls smoothness)
        """
        self.module = module
        self.teeth_number = teeth_number
        self.pressure_angle = pressure_angle
        self.cone_angle = cone_angle
        self.face_width = face_width
        self.clearance = clearance
        self.backlash = backlash
        self.profile_shift = profile_shift
        self.frame_count = frame_count
        
        # Calculate tooth width from module
        self.tooth_width = np.pi * module
        
        # Calculate pitch radius at the large end of the cone
        self.pitch_radius = (teeth_number * module) / 2
        
        # Generate gear profiles at different cone sections
        self.sections = self._generate_sections()
    
    def _generate_sections(self, num_sections=10):
        """Generate gear profiles at different sections along the cone"""
        sections = []
        
        # Calculate the cone apex distance
        apex_distance = self.pitch_radius / math.sin(math.radians(self.cone_angle))
        
        # Generate sections from the large end to the small end
        for i in range(num_sections):
            # Position along the cone (0 = large end, 1 = small end)
            t = i / (num_sections - 1)
            
            # Distance from the apex to this section
            section_distance = apex_distance - t * self.face_width
            
            # Scale factor for this section
            scale = section_distance / apex_distance
            
            # Effective module and pitch radius at this section
            effective_module = self.module * scale
            effective_radius = self.pitch_radius * scale
            
            # Generate gear profile for this section
            if scale > 0.1:  # Avoid too small sections
                tooth_width = np.pi * effective_module
                gear_poly, _ = gear.generate(
                    teeth_count=self.teeth_number,
                    tooth_width=tooth_width,
                    pressure_angle=gear.deg2rad(self.pressure_angle),
                    backlash=self.backlash,
                    frame_count=self.frame_count
                )
                
                # Get points and scale them
                points = np.array(gear_poly.exterior.coords)
                
                # Calculate z-coordinate for this section
                z = t * self.face_width
                
                # Add section data
                sections.append({
                    'points': points,
                    'z': z,
                    'scale': scale,
                    'radius': effective_radius
                })
        
        return sections

    def get_3d_points(self):
        """Get 3D points for the gear"""
        all_points = []
        
        for section in self.sections:
            points_2d = section['points']
            z = section['z']
            
            # Convert 2D points to 3D
            points_3d = np.column_stack((points_2d, np.full(len(points_2d), z)))
            all_points.append(points_3d)
            
        return all_points

def create_rhino_bevel_gear(gear, hole_radius=5.0):
    """Create a Rhino 3D model of the bevel gear"""
    if not RHINO_AVAILABLE:
        print("Error: rhinoinside module not available. Cannot create 3DM file.")
        return None
    
    # Get 3D points for all sections
    sections = gear.get_3d_points()
    
    # Create curves for each section
    curves = []
    for section_points in sections:
        # Create a closed curve for this section
        points3d = [rg.Point3d(p[0], p[1], p[2]) for p in section_points]
        curve = rg.Curve.CreateInterpolatedCurve(points3d, 3)
        if curve:
            curves.append(curve)
    
    # Create loft through all section curves
    brep = None
    if len(curves) >= 2:
        loft_type = rg.LoftType.Normal
        loft = rg.Brep.CreateFromLoft(curves, rg.Point3d.Unset, rg.Point3d.Unset, 
                                      loft_type, False)
        if loft and len(loft) > 0:
            brep = loft[0]
    
    # Create center hole
    if brep:
        # Create a cylinder for the center hole
        circle = rg.Circle(rg.Plane.WorldXY, rg.Point3d(0, 0, 0), hole_radius)
        cylinder = rg.Cylinder(circle, gear.face_width)
        cylinder_brep = rg.Brep.CreateFromCylinder(cylinder, True, True)
        
        # Boolean difference to create the hole
        if cylinder_brep:
            result = rg.Brep.CreateBooleanDifference([brep], [cylinder_brep], 0.01)
            if result and len(result) > 0:
                brep = result[0]
    
    return brep

def export_bevel_gears(module, teeth1, teeth2, pressure_angle, cone_angle, face_width, 
                       clearance, backlash, filename, frame_count):
    """
    Create and export bevel gears to a 3DM file
    
    Parameters:
    -----------
    module : float
        Module (tooth size unit)
    teeth1, teeth2 : int
        Number of teeth for each gear
    pressure_angle : float
        Pressure angle in degrees
    cone_angle : float
        Cone angle in degrees
    face_width : float
        Width of the gear face
    clearance, backlash : float
        Gear parameters
    filename : str
        Output filename (.3dm)
    frame_count : int
        Number of points per curve
    """
    if not RHINO_AVAILABLE:
        print("Error: rhinoinside module not available. Cannot create 3DM file.")
        return
    
    # Create the first bevel gear
    gear1 = BevelGear(
        module=module,
        teeth_number=teeth1,
        pressure_angle=pressure_angle,
        cone_angle=cone_angle,
        face_width=face_width,
        clearance=clearance,
        backlash=backlash,
        frame_count=frame_count
    )
    
    # Create the second bevel gear
    gear2 = BevelGear(
        module=module,
        teeth_number=teeth2,
        pressure_angle=pressure_angle,
        cone_angle=90 - cone_angle,  # Complementary angle for 90° mating
        face_width=face_width,
        clearance=clearance,
        backlash=backlash,
        frame_count=frame_count
    )
    
    # Create Rhino 3D models
    brep1 = create_rhino_bevel_gear(gear1, hole_radius=module*2)
    brep2 = create_rhino_bevel_gear(gear2, hole_radius=module*2)
    
    # Position the second gear (rotate and translate)
    if brep2:
        # Create rotation transform (90 degrees around Y-axis)
        rotation = rg.Transform.Rotation(math.pi/2, rg.Vector3d(0, 1, 0), rg.Point3d(0, 0, 0))
        
        # Calculate translation to position at the apex
        apex_distance1 = gear1.pitch_radius / math.sin(math.radians(gear1.cone_angle))
        translation = rg.Transform.Translation(apex_distance1, 0, 0)
        
        # Apply transforms
        brep2.Transform(rotation)
        brep2.Transform(translation)
    
    # Create a new Rhino file
    file3dm = Rhino.FileIO.File3dm()
    
    # Add the gears to the file
    if brep1:
        attr1 = Rhino.DocObjects.ObjectAttributes()
        attr1.Name = f"Bevel Gear {teeth1} teeth"
        attr1.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
        attr1.ObjectColor = System.Drawing.Color.LightGray
        file3dm.Objects.Add(brep1, attr1)
    
    if brep2:
        attr2 = Rhino.DocObjects.ObjectAttributes()
        attr2.Name = f"Bevel Gear {teeth2} teeth"
        attr2.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
        attr2.ObjectColor = System.Drawing.Color.DarkGray
        file3dm.Objects.Add(brep2, attr2)
    
    # Save the file
    file3dm.Write(filename, 7)  # Version 7 format
    print(f"Bevel gears exported to {filename}")

def visualize_bevel_gears(gear1, gear2):
    """Visualize the bevel gears in 3D using Matplotlib"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the first gear
    sections1 = gear1.get_3d_points()
    for section in sections1:
        x, y, z = section[:, 0], section[:, 1], section[:, 2]
        ax.plot(x, y, z, 'b-', alpha=0.5, linewidth=0.5)
    
    # Calculate apex distance and rotation for the second gear
    apex_distance = gear1.pitch_radius / math.sin(math.radians(gear1.cone_angle))
    
    # Plot the second gear (rotated 90 degrees)
    sections2 = gear2.get_3d_points()
    rot = R.from_euler('y', 90, degrees=True)
    
    for section in sections2:
        # Rotate points
        rotated = rot.apply(section)
        
        # Translate to position
        rotated[:, 0] += apex_distance
        
        x, y, z = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        ax.plot(x, y, z, 'r-', alpha=0.5, linewidth=0.5)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bevel Gears (90° Mating)')
    
    # Set limits with some margin
    max_radius = max(gear1.pitch_radius, gear2.pitch_radius)
    max_width = max(gear1.face_width, gear2.face_width)
    limit = max(max_radius, apex_distance) * 1.2
    
    ax.set_xlim([-limit/2, limit + limit/2])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    
    return fig, ax

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Generate and visualize bevel gears for 90° mating')
    parser.add_argument('--module', type=float, default=5.0, help='Module (tooth size unit)')
    parser.add_argument('--teeth1', type=int, default=20, help='Number of teeth for the first gear')
    parser.add_argument('--teeth2', type=int, default=20, help='Number of teeth for the second gear')
    parser.add_argument('--pressure-angle', type=float, default=20.0, help='Pressure angle in degrees')
    parser.add_argument('--cone-angle', type=float, default=45.0, help='Cone angle in degrees (typically 45° for 90° mating)')
    parser.add_argument('--face-width', type=float, default=15.0, help='Width of the gear face')
    parser.add_argument('--backlash', type=float, default=0.0, help='Backlash')
    parser.add_argument('--clearance', type=float, default=0.25, help='Clearance coefficient')
    parser.add_argument('--save', type=str, help='Save 3D model to file (.3dm)')
    parser.add_argument('--save-image', type=str, help='Save visualization to image file')
    parser.add_argument('--frame-count', type=int, default=16, help='Number of points per curve (lower values = fewer vertices)')
    parser.add_argument('--show', type=bool, default=True, help='Show the visualization')
    args = parser.parse_args()

    # Create the first bevel gear
    gear1 = BevelGear(
        module=args.module,
        teeth_number=args.teeth1,
        pressure_angle=args.pressure_angle,
        cone_angle=args.cone_angle,
        face_width=args.face_width,
        clearance=args.clearance,
        backlash=args.backlash,
        frame_count=args.frame_count
    )
    
    # Create the second bevel gear
    gear2 = BevelGear(
        module=args.module,
        teeth_number=args.teeth2,
        pressure_angle=args.pressure_angle,
        cone_angle=90 - args.cone_angle,  # Complementary angle for 90° mating
        face_width=args.face_width,
        clearance=args.clearance,
        backlash=args.backlash,
        frame_count=args.frame_count
    )
    
    # Visualize the gears
    if args.show or args.save_image:
        fig, ax = visualize_bevel_gears(gear1, gear2)
        
        # Save visualization if requested
        if args.save_image:
            plt.savefig(args.save_image, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {args.save_image}")
        
        # Show the visualization if requested
        if args.show:
            plt.show()
    
    # Export 3D model if requested
    if args.save:
        if not args.save.lower().endswith('.3dm'):
            args.save += '.3dm'
        
        export_bevel_gears(
            args.module,
            args.teeth1,
            args.teeth2,
            args.pressure_angle,
            args.cone_angle,
            args.face_width,
            args.clearance,
            args.backlash,
            args.save,
            args.frame_count
        )

if __name__ == "__main__":
    main()