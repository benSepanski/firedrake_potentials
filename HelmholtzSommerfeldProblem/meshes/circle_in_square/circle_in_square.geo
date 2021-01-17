SetFactory("OpenCASCADE");
Merge "circle_in_square.brep";

// This line (and the SetFactory) merge duplicate lines created by CAD,
// very imporant!
Coherence;

//+
Physical Line("scatterer") = {11};
//+
Physical Line("outer_bdy") = {18, 13, 14, 8, 25, 24, 20, 22, 23, 2, 16, 15};
//+
Physical Surface("inner_region") = {3};
//+
Physical Surface("pml_x_region") = {2, 1};
//+
Physical Surface("pml_y_region") = {6, 7};
//+
Physical Surface("pml_xy_region") = {5, 8, 9, 4};
