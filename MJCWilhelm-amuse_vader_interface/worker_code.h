
int set_outer_pressure_boundary_mass_flux(double obc_pres_val);

int get_time(double * time);

int set_delta(double delta);

int get_inner_pressure_boundary_torque(double * ibc_pres_val);

int set_inner_boundary_function(int ibc_func);

int get_grid_pressure(int * i, double * pressure, int number_of_points);

int set_inner_pressure_boundary_mass_flux(double ibc_pres_val);

int get_parameter(int i, double * param);

int get_maxIter(int * maxIter);

int get_outer_pressure_boundary_torque_flux(double * obc_pres_val);

int get_inner_pressure_boundary_mass_flux(double * ibc_pres_val);

int get_inner_enthalpy_boundary_type(int * ibc_enth);

int get_outer_pressure_boundary_mass_flux(double * obc_pres_val);

int get_delta(double * delta);

int update_keplerian_grid(double vphi);

int set_grid_internal_energy(int * i, double * internal_energy, int number_of_points);

int get_gravitational_potential_of_index(int * i, double * psi_grav, int number_of_points);

int initialize_keplerian_grid(int n, int linear, double rmin, double rmax, double m);

int get_outer_pressure_boundary_torque(double * obc_pres_val);

int get_inner_boundary_mass_out(double * mSrcOut);

int get_errTol(double * errTol);

int get_number_of_cells(int * i);

int set_maxIter(int maxIter);

int get_inner_enthalpy_boundary_enthalpy_gradient(double * ibc_enth_val);

int set_internal_energy_source_function(int internal_energy_source_func);

int get_eos_function(int * eos_func);

int set_dtTol(double dtTol);

int get_rotational_velocity_of_index(int * i, double * vphi, int number_of_points);

int initialize_flat_grid(int n, int linear, double rmin, double rmax, double vphi);

int evolve_model(double tlim);

int set_number_of_user_parameters(int n);

int get_dtTol(double * dtTol);

int get_PreTimestep(int * PreTimestep);

int get_outer_enthalpy_boundary_enthalpy_gradient(double * obc_enth_val);

int get_grid_internal_energy(int * i, double * internal_energy, int number_of_points);

int get_mass_source_value(double * mass_source_value);

int set_inner_pressure_boundary_type(int ibc_pres);

int set_outer_pressure_boundary_type(int obc_pres);

int get_grid_state(int * i, double * sigma, double * pressure, double * internal_energy, int number_of_points);

int set_verbosity(int verbosity);

int set_outer_pressure_boundary_torque_flux(double obc_pres_val);

int get_inner_boundary_energy_out(double * eSrcOut);

int set_eos_function(int eos_func);

int get_maxStep(int * maxStep);

int get_area_of_index(int * i, double * area, int number_of_points);

int set_PreTimestep(int PreTimestep);

int get_internal_energy_source_function(int * internal_energy_source_func);

int set_inner_enthalpy_boundary_enthalpy_gradient(double ibc_enth_val);

int get_outer_enthalpy_boundary_type(int * obc_enth);

int get_number_of_user_parameters(int * n);

int set_maxStep(int maxStep);

int set_internal_energy_source_value(double internal_energy_source_value);

int set_outer_enthalpy_boundary_enthalpy_gradient(double obc_enth_val);

int get_inner_boundary_function(int * ibc_func);

int get_maxDtIncrease(double * maxDtIncrease);

int set_errTol(double errTol);

int get_alpha_function(int * alpha_func);

int set_mass_source_value(double mass_source_value);

int set_dtMin(double dtMin);

int set_alpha(double alpha);

int get_mass_source_function(int * mass_source_func);

int get_index_of_position(double r, int * i);

int set_grid_column_density(int * i, double * sigma, int number_of_points);

int set_inner_pressure_boundary_torque(double ibc_pres_val);

int get_alpha(double * alpha);

int get_grid_column_density(int * i, double * column_density, int number_of_points);

int get_dtMin(double * dtMin);

int set_outer_boundary_function(int obc_func);

int get_mass_source_out(int * i, double * mSrcOut, int number_of_points);

int get_outer_pressure_boundary_type(int * obc_pres);

int set_PostTimestep(int PostTimestep);

int set_interpOrder(int interpOrder);

int set_gamma(double gamma);

int set_mass_source_function(int mass_source_func);

int get_internal_energy_source_value(double * internal_energy_source_value);

int get_inner_pressure_boundary_torque_flux(double * ibc_pres_val);

int get_effective_potential_of_index(int * i, double * psiEff, int number_of_points);

int set_inner_enthalpy_boundary_enthalpy(double ibc_enth_val);

int get_outer_boundary_energy_out(double * eSrcOut);

int get_verbosity(int * verbosity);

int set_outer_enthalpy_boundary_type(int obc_enth);

int set_maxDtIncrease(double maxDtIncrease);

int set_outer_enthalpy_boundary_enthalpy(double obc_enth_val);

int set_outer_pressure_boundary_torque(double obc_pres_val);

int set_grid_state(int * i, double * sigma, double * pressure, double * internal_energy, int number_of_points);

int get_outer_boundary_mass_out(double * mSrcOut);

int get_gamma(double * gamma);

int set_grid_pressure(int * i, double * pressure, int number_of_points);

int get_useBE(int * useBE);

int get_interpOrder(int * interpOrder);

int cleanup_code();

int set_alpha_function(int alpha_func);

int set_inner_enthalpy_boundary_type(int ibc_enth);

int recommit_parameters();

int initialize_code();

int set_dtStart(double dtStart);

int get_position_of_index(int i, double * r);

int set_useBE(int useBE);

int get_inner_enthalpy_boundary_enthalpy(double * ibc_enth_val);

int get_energy_source_out(int * i, double * eSrcOut, int number_of_points);

int get_outer_boundary_function(int * obc_func);

int get_inner_pressure_boundary_type(int * ibc_pres);

int set_parameter(int i, double param);

int get_dtStart(double * dtStart);

int set_inner_pressure_boundary_torque_flux(double ibc_pres_val);

int commit_parameters();

int get_PostTimestep(int * PostTimestep);

int get_outer_enthalpy_boundary_enthalpy(double * obc_enth_val);

