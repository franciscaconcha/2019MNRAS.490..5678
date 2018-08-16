
#ifndef NOMPI
    #include <mpi.h>
#endif
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef WIN32
	#include <winsock2.h>
#else
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <netdb.h>
	#include <unistd.h>
	#include <netinet/tcp.h>
  #include <arpa/inet.h>
#endif

#include "worker_code.h"

static bool NEEDS_MPI = true;

static int MAX_INTS_IN = 1;
static int MAX_INTS_OUT = 2;
static int MAX_STRINGS_IN = 2;
static int MAX_STRINGS_OUT = 1;
static int MAX_DOUBLES_IN = 3;
static int MAX_DOUBLES_OUT = 3;
static int MAX_LONGS_IN = 0;
static int MAX_LONGS_OUT = 0;
static int MAX_BOOLEANS_IN = 1;
static int MAX_BOOLEANS_OUT = 1;
static int MAX_FLOATS_IN = 0;
static int MAX_FLOATS_OUT = 0;


static int ERROR_FLAG = 256;
static int HEADER_SIZE = 11; //integers

static int HEADER_FLAGS = 0;
static int HEADER_CALL_ID = 1;
static int HEADER_FUNCTION_ID = 2;
static int HEADER_CALL_COUNT = 3;
static int HEADER_INTEGER_COUNT = 4;
static int HEADER_LONG_COUNT = 5;
static int HEADER_FLOAT_COUNT = 6;
static int HEADER_DOUBLE_COUNT = 7;
static int HEADER_BOOLEAN_COUNT = 8;
static int HEADER_STRING_COUNT = 9;
static int HEADER_UNITS_COUNT = 10;

static bool TRUE_BYTE = 1;
static bool FALSE_BYTE = 0;

static bool mpiIntercom = false;

static int socketfd = 0;

static int * header_in;
static int * header_out;

static int * ints_in;
static int * ints_out;

static long long int * longs_in;
static long long int * longs_out;

static float * floats_in;
static float * floats_out;

static double * doubles_in;
static double * doubles_out;

static int * booleans_in;
static int * booleans_out;

/* sizes of strings */
static int * string_sizes_in;
static int * string_sizes_out;

/* pointers to input and output strings (contents not stored here) */
static char * * strings_in;
static char * * strings_out;

/* actual string data */
static char * characters_in = 0;
static char * characters_out = 0;


static int polling_interval = 0;
#ifndef NOMPI
#define MAX_COMMUNICATORS 2048
static char portname_buffer[MPI_MAX_PORT_NAME+1];
static MPI_Comm communicators[MAX_COMMUNICATORS];
static int lastid = -1;
static int activeid = -1;
static int id_to_activate = -1;
#else
static const char * empty_string = "";
#endif

int internal__get_message_polling_interval(int * outval)
{
    *outval = polling_interval;
    
    return 0;
}

int internal__set_message_polling_interval(int inval)
{
    polling_interval = inval;
    
    return 0;
}
int internal__open_port(char ** output)
{
#ifndef NOMPI
    MPI_Open_port(MPI_INFO_NULL, portname_buffer);
    *output = portname_buffer;
#else
    *output = (char *) empty_string;
#endif
    return 0;
}
int internal__accept_on_port(char * port_identifier, int * comm_identifier)
{
#ifndef NOMPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    lastid++;
    if(lastid >= MAX_COMMUNICATORS) {
        lastid--;
        return -1;
    }
    if(rank == 0){
        MPI_Comm merged;
        MPI_Comm communicator;
        MPI_Comm_accept(port_identifier, MPI_INFO_NULL, 0,  MPI_COMM_SELF, &communicator);
        MPI_Intercomm_merge(communicator, 0, &merged);
        MPI_Intercomm_create(MPI_COMM_WORLD,0,merged, 1, 65, &communicators[lastid]);
        MPI_Comm_disconnect(&merged);
        MPI_Comm_disconnect(&communicator);
    } else {
        MPI_Intercomm_create(MPI_COMM_WORLD,0, MPI_COMM_NULL, 1, 65, &communicators[lastid]);
    }
    *comm_identifier = lastid;
#else
    *comm_identifier = -1;
#endif
    return 0;
}


int internal__connect_to_port(char * port_identifier, int * comm_identifier)
{
#ifndef NOMPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    lastid++;
    if(lastid >= MAX_COMMUNICATORS) {
        lastid--;
        return -1;
    }
    if(rank == 0){
        MPI_Comm merged;
        MPI_Comm communicator;
        MPI_Comm_connect(port_identifier, MPI_INFO_NULL, 0,  MPI_COMM_SELF, &communicator);
        MPI_Intercomm_merge(communicator, 1, &merged);
        MPI_Intercomm_create(MPI_COMM_WORLD, 0, merged, 0, 65, &communicators[lastid]);
        MPI_Comm_disconnect(&merged);
        MPI_Comm_disconnect(&communicator);
    } else {
        MPI_Intercomm_create(MPI_COMM_WORLD, 0, MPI_COMM_NULL, 1, 65, &communicators[lastid]);
    }
    *comm_identifier = lastid;
#else
    *comm_identifier = -1;
#endif
    return 0;
}

int internal__activate_communicator(int comm_identifier){
#ifndef NOMPI
    if(comm_identifier < 0 || comm_identifier > lastid) {
        return -1;
    }
    id_to_activate = comm_identifier;
#endif
    return 0;
}

int internal__become_code(int number_of_workers, char * modulename, char * classname)
{
    return 0;
}



#ifndef NOMPI
#include <unistd.h>

int mpi_recv_header(MPI_Comm & parent)
{
    MPI_Request header_request;
    MPI_Status request_status;
   
    MPI_Irecv(header_in, HEADER_SIZE, MPI_INT, 0, 989, parent, &header_request);
        
    if(polling_interval > 0)
    {
        int is_finished = 0;
        MPI_Test(&header_request, &is_finished, &request_status);
        while(!is_finished) {
            usleep(polling_interval);
            MPI_Test(&header_request, &is_finished, &request_status);
        }
        MPI_Wait(&header_request, &request_status);
    } else {
        MPI_Wait(&header_request, &request_status);
    }
    return 0;
}
#endif


bool handle_call() {
  int call_count = header_in[HEADER_CALL_COUNT];
  
  switch(header_in[HEADER_FUNCTION_ID]) {
    case 0:
      return false;
      break;
    case 5659179:
      ints_out[0] = set_outer_pressure_boundary_mass_flux(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 44188957:
      ints_out[0] = get_time(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 45220543:
      ints_out[0] = set_delta(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 46710460:
      ints_out[0] = get_inner_pressure_boundary_torque(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 62907364:
      ints_out[0] = set_inner_boundary_function(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 79858498:
      ints_out[0] = get_grid_pressure(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 94728249:
      ints_out[0] = set_inner_pressure_boundary_mass_flux(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 109050006:
      ints_out[0] = get_parameter(
        ints_in[0] ,
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 136213299:
      ints_out[0] = get_maxIter(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 153650193:
      ints_out[0] = get_outer_pressure_boundary_torque_flux(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 164264507:
      ints_out[0] = get_inner_pressure_boundary_mass_flux(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 193293590:
      ints_out[0] = get_inner_enthalpy_boundary_type(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 205082665:
      ints_out[0] = get_outer_pressure_boundary_mass_flux(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 207744574:
      ints_out[0] = get_delta(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 226141313:
      ints_out[0] = update_keplerian_grid(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 264160343:
      ints_out[0] = set_grid_internal_energy(
        &ints_in[0] ,
        &doubles_in[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 275646056:
      ints_out[0] = get_gravitational_potential_of_index(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 284214138:
      ints_out[0] = initialize_keplerian_grid(
        ints_in[0] ,
        booleans_in[0] ,
        doubles_in[0] ,
        doubles_in[1] ,
        doubles_in[2]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 319624261:
      ints_out[0] = get_outer_pressure_boundary_torque(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 321410708:
      ints_out[0] = get_inner_boundary_mass_out(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 334812026:
      ints_out[0] = get_errTol(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 351359763:
      ints_out[0] = get_number_of_cells(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 372360316:
      ints_out[0] = set_maxIter(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 383112453:
      ints_out[0] = internal__connect_to_port(
        strings_in[0] ,
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 388118932:
      ints_out[0] = get_inner_enthalpy_boundary_enthalpy_gradient(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 393252046:
      ints_out[0] = set_internal_energy_source_function(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 397993125:
      ints_out[0] = get_eos_function(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 401092938:
      ints_out[0] = set_dtTol(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 407982351:
      ints_out[0] = get_rotational_velocity_of_index(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 416314728:
      ints_out[0] = initialize_flat_grid(
        ints_in[0] ,
        booleans_in[0] ,
        doubles_in[0] ,
        doubles_in[1] ,
        doubles_in[2]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 416721977:
      ints_out[0] = evolve_model(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 420390650:
      ints_out[0] = set_number_of_user_parameters(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 423353801:
      ints_out[0] = get_dtTol(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 423557428:
      ints_out[0] = get_PreTimestep(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 429362641:
      ints_out[0] = get_outer_enthalpy_boundary_enthalpy_gradient(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 433268021:
      ints_out[0] = get_grid_internal_energy(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 448677469:
      ints_out[0] = get_mass_source_value(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 510964419:
      ints_out[0] = set_inner_pressure_boundary_type(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 560213702:
      ints_out[0] = set_outer_pressure_boundary_type(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 572989170:
      ints_out[0] = get_grid_state(
        &ints_in[0] ,
        &doubles_out[0] ,
        &doubles_out[( 1 * call_count)] ,
        &doubles_out[( 2 * call_count)] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 3 * call_count;
      break;
    
    case 599613284:
      ints_out[0] = set_verbosity(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 607452190:
      ints_out[0] = set_outer_pressure_boundary_torque_flux(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 614147446:
      ints_out[0] = get_inner_boundary_energy_out(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 632198846:
      ints_out[0] = set_eos_function(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 642472474:
      ints_out[0] = get_maxStep(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 683512808:
      ints_out[0] = get_area_of_index(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 685767904:
      ints_out[0] = set_PreTimestep(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 727361823:
      ints_out[0] = internal__set_message_polling_interval(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 804218075:
      ints_out[0] = get_internal_energy_source_function(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 853200593:
      ints_out[0] = set_inner_enthalpy_boundary_enthalpy_gradient(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 882241809:
      ints_out[0] = get_outer_enthalpy_boundary_type(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 919768251:
      ints_out[0] = internal__get_message_polling_interval(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 928495314:
      ints_out[0] = get_number_of_user_parameters(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 946138451:
      ints_out[0] = set_maxStep(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 969740198:
      ints_out[0] = set_internal_energy_source_value(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1013938834:
      ints_out[0] = set_outer_enthalpy_boundary_enthalpy_gradient(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1015686132:
      ints_out[0] = get_inner_boundary_function(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 1021734878:
      ints_out[0] = get_maxDtIncrease(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1035072236:
      ints_out[0] = set_errTol(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1081588996:
      ints_out[0] = get_alpha_function(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 1134776330:
      ints_out[0] = set_mass_source_value(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1135907041:
      ints_out[0] = set_dtMin(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1142015244:
      ints_out[0] = set_alpha(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1152277666:
      ints_out[0] = get_mass_source_function(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 1211600737:
      for (int i = 0 ; i < call_count; i++){
        ints_out[i] = get_index_of_position(
          doubles_in[i] ,
          &ints_out[( 1 * call_count) + i]
        );
      }
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 1250028876:
      ints_out[0] = set_grid_column_density(
        &ints_in[0] ,
        &doubles_in[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1250434105:
      ints_out[0] = set_inner_pressure_boundary_torque(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1254238607:
      ints_out[0] = get_alpha(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1269811859:
      ints_out[0] = get_grid_column_density(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1298627682:
      ints_out[0] = get_dtMin(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1298877048:
      ints_out[0] = set_outer_boundary_function(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1302461185:
      ints_out[0] = get_mass_source_out(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1303861767:
      ints_out[0] = get_outer_pressure_boundary_type(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 1309356817:
      ints_out[0] = set_PostTimestep(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1346197981:
      ints_out[0] = set_interpOrder(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1353853975:
      ints_out[0] = set_gamma(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1388559812:
      ints_out[0] = set_mass_source_function(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1428061031:
      ints_out[0] = get_internal_energy_source_value(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1436075801:
      ints_out[0] = get_inner_pressure_boundary_torque_flux(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1444911473:
      ints_out[0] = get_effective_potential_of_index(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1445591169:
      ints_out[0] = set_inner_enthalpy_boundary_enthalpy(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1449506300:
      ints_out[0] = get_outer_boundary_energy_out(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1459852080:
      ints_out[0] = get_verbosity(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 1481016788:
      ints_out[0] = set_outer_enthalpy_boundary_type(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1488965996:
      ints_out[0] = set_maxDtIncrease(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1513820140:
      ints_out[0] = internal__become_code(
        ints_in[0] ,
        strings_in[0] ,
        strings_in[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1521157816:
      ints_out[0] = set_outer_enthalpy_boundary_enthalpy(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1531819714:
      ints_out[0] = set_outer_pressure_boundary_torque(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1564061061:
      ints_out[0] = set_grid_state(
        &ints_in[0] ,
        &doubles_in[0] ,
        &doubles_in[( 1 * call_count)] ,
        &doubles_in[( 2 * call_count)] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1576861448:
      ints_out[0] = get_outer_boundary_mass_out(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1583464598:
      ints_out[0] = get_gamma(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1620784628:
      ints_out[0] = set_grid_pressure(
        &ints_in[0] ,
        &doubles_in[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1638066584:
      ints_out[0] = get_useBE(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 1641831439:
      ints_out[0] = get_interpOrder(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 1644113439:
      ints_out[0] = cleanup_code();
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1702639130:
      ints_out[0] = set_alpha_function(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1706735752:
      ints_out[0] = internal__activate_communicator(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1733695957:
      ints_out[0] = set_inner_enthalpy_boundary_type(
        ints_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1744145122:
      ints_out[0] = recommit_parameters();
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1768994498:
      ints_out[0] = initialize_code();
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1829802798:
      ints_out[0] = set_dtStart(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1829880540:
      ints_out[0] = internal__open_port(
        &strings_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_STRING_COUNT] = 1 * call_count;
      break;
    
    case 1853731497:
      for (int i = 0 ; i < call_count; i++){
        ints_out[i] = get_position_of_index(
          ints_in[i] ,
          &doubles_out[i]
        );
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1869677851:
      ints_out[0] = set_useBE(
        booleans_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1891926464:
      ints_out[0] = internal__accept_on_port(
        strings_in[0] ,
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 1892105652:
      ints_out[0] = get_inner_enthalpy_boundary_enthalpy(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1915891005:
      ints_out[0] = get_energy_source_out(
        &ints_in[0] ,
        &doubles_out[0] ,
        call_count
      );
      for (int i = 1 ; i < call_count; i++){
        ints_out[i] = ints_out[0];
      }
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 1918798440:
      ints_out[0] = get_outer_boundary_function(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 1923597824:
      ints_out[0] = get_inner_pressure_boundary_type(
        &ints_out[1]
      );
      header_out[HEADER_INTEGER_COUNT] = 2 * call_count;
      break;
    
    case 1925264602:
      ints_out[0] = set_parameter(
        ints_in[0] ,
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 1933517927:
      ints_out[0] = get_dtStart(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    case 2021965588:
      ints_out[0] = set_inner_pressure_boundary_torque_flux(
        doubles_in[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 2069478464:
      ints_out[0] = commit_parameters();
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      break;
    
    case 2082291978:
      ints_out[0] = get_PostTimestep(
        &booleans_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_BOOLEAN_COUNT] = 1 * call_count;
      break;
    
    case 2084946829:
      ints_out[0] = get_outer_enthalpy_boundary_enthalpy(
        &doubles_out[0]
      );
      header_out[HEADER_INTEGER_COUNT] = 1 * call_count;
      header_out[HEADER_DOUBLE_COUNT] = 1 * call_count;
      break;
    
    default:
      header_out[HEADER_FLAGS] = header_out[HEADER_FLAGS] | ERROR_FLAG;
      strings_out[0] = new char[100];
      sprintf(strings_out[0], "unknown function id: %d\n", header_in[HEADER_FUNCTION_ID]);
      fprintf(stderr, "unknown function id: %d\n", header_in[HEADER_FUNCTION_ID]);
      header_out[HEADER_STRING_COUNT] = 1;
  }
  return true;
}

void onexit_mpi(void) {
#ifndef NOMPI
    int flag = 0;
    MPI_Finalized(&flag);
    
    if(!flag) {
        MPI_Comm parent;
        MPI_Comm_get_parent(&parent);
        
        int rank = 0;
        
        MPI_Comm_rank(parent, &rank);
        
        header_out[HEADER_FLAGS] = ERROR_FLAG;

        header_out[HEADER_CALL_ID] = 0;
        header_out[HEADER_FUNCTION_ID] = 0;
        header_out[HEADER_CALL_COUNT] = 0;
        header_out[HEADER_INTEGER_COUNT] = 0;
        header_out[HEADER_LONG_COUNT] = 0;
        header_out[HEADER_FLOAT_COUNT] = 0;
        header_out[HEADER_DOUBLE_COUNT] = 0;
        header_out[HEADER_BOOLEAN_COUNT] = 0;
        header_out[HEADER_STRING_COUNT] = 0;
        header_out[HEADER_UNITS_COUNT] = 0;

        MPI_Send(header_out, HEADER_SIZE, MPI_INT, 0, 999, parent);
        
        for(int i = 0; i < lastid + 1; i++) {
            MPI_Comm_disconnect(&communicators[i]);
        }
        
        MPI_Finalize();
    }
#endif
}

void onexit_sockets(void) {
#ifdef WIN32
	closesocket(socketfd);
#else
	close(socketfd);
#endif
}

void send_array_sockets(void *buffer, int length, int file_descriptor, int rank) {
    int total_written = 0;
    int bytes_written;

    if (rank != 0) {
        return;
    }
    //fprintf(stderr, "number of bytes to write: %d\n", length);
    while (total_written < length) {
    
#ifdef WIN32
        bytes_written = send(file_descriptor, ((char *) buffer) + total_written,
                        length - total_written, 0);
#else
        bytes_written = write(file_descriptor, ((char *) buffer) + total_written,
                        length - total_written);
#endif


        if (bytes_written == -1) {
            perror("could not write data");
            exit(1);
        }

        total_written = total_written + bytes_written;
    }
}

void receive_array_sockets(void *buffer, int length, int file_descriptor, int rank) {
    int total_read = 0;
    int bytes_read;

    if (rank != 0) {
        return;
    }

    while (total_read < length) {
    
#ifdef WIN32
        bytes_read = recv(file_descriptor, ((char *) buffer) + total_read,
                        length - total_read, 0);
#else
        bytes_read = read(file_descriptor, ((char *) buffer) + total_read,
                        length - total_read);
#endif

        if (bytes_read == -1) {
            perror("could not read data");
            exit(1);
        }

        total_read = total_read + bytes_read;
    }
}

void new_arrays(int max_call_count) {
  ints_in = new int[ max_call_count * MAX_INTS_IN];
  ints_out = new int[ max_call_count * MAX_INTS_OUT];

  longs_in = new long long int[ max_call_count * MAX_LONGS_IN];
  longs_out = new long long int[ max_call_count * MAX_LONGS_OUT];

  floats_in = new float[ max_call_count * MAX_FLOATS_IN];
  floats_out = new float[ max_call_count * MAX_FLOATS_OUT];

  doubles_in = new double[ max_call_count * MAX_DOUBLES_IN];
  doubles_out = new double[ max_call_count * MAX_DOUBLES_OUT];
  
  booleans_in = new int[ max_call_count * MAX_BOOLEANS_IN];
  booleans_out = new int[ max_call_count * MAX_BOOLEANS_OUT];
  
  string_sizes_in = new int[ max_call_count * MAX_STRINGS_IN];
  string_sizes_out = new int[ max_call_count * MAX_STRINGS_OUT];

  strings_in = new char *[ max_call_count * MAX_STRINGS_IN];
  strings_out = new char *[ max_call_count * MAX_STRINGS_OUT];
}

void delete_arrays() {
  delete[] ints_in;
  delete[] ints_out;
  delete[] longs_in;
  delete[] longs_out;
  delete[] floats_in;
  delete[] floats_out;
  delete[] doubles_in;
  delete[] doubles_out;
  delete[] booleans_in;
  delete[] booleans_out;
  delete[] string_sizes_in;
  delete[] string_sizes_out;
  delete[] strings_in;
  delete[] strings_out;
}

void run_mpi(int argc, char *argv[]) {
#ifndef NOMPI
  int provided;
  int rank = 0;
  
  mpiIntercom = true;

  //fprintf(stderr, "C worker: running in mpi mode\n");
  
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm parent;
  MPI_Comm_get_parent(&communicators[0]);
  lastid += 1;
  activeid = 0;
  parent = communicators[activeid];
  MPI_Comm_rank(parent, &rank);
  atexit(onexit_mpi);
  
  bool must_run_loop = true;
  
  int max_call_count = 10;
  
  header_in = new int[HEADER_SIZE];
  header_out = new int[HEADER_SIZE];

  new_arrays(max_call_count);  

  while(must_run_loop) {
    //fprintf(stderr, "receiving header\n");
    if(id_to_activate >= 0 && id_to_activate != activeid){
        activeid = id_to_activate;
        id_to_activate = -1;
        parent = communicators[activeid];
        MPI_Comm_rank(parent, &rank);
    }
    
    mpi_recv_header(parent);
    
    //fprintf(stderr, "C worker code: got header %d %d %d %d %d %d %d %d %d %d\n", header_in[0], header_in[1], header_in[2], header_in[3], header_in[4], header_in[5], header_in[6], header_in[7], header_in[8], header_in[9]);
    
    int call_count = header_in[HEADER_CALL_COUNT];

    if (call_count > max_call_count) {
      delete_arrays();
      max_call_count = call_count + 255;
      new_arrays(max_call_count);
    }
    
    if(header_in[HEADER_INTEGER_COUNT] > 0) {
      MPI_Bcast(ints_in, header_in[HEADER_INTEGER_COUNT] , MPI_INT, 0, parent);
    }
    
    if(header_in[HEADER_LONG_COUNT] > 0) {
      MPI_Bcast(longs_in, header_in[HEADER_LONG_COUNT], MPI_LONG_LONG_INT, 0, parent);
    }
    
    if(header_in[HEADER_FLOAT_COUNT] > 0) {
      MPI_Bcast(floats_in, header_in[HEADER_FLOAT_COUNT], MPI_FLOAT, 0, parent);
    }
    
    if(header_in[HEADER_DOUBLE_COUNT] > 0) {
      MPI_Bcast(doubles_in, header_in[HEADER_DOUBLE_COUNT], MPI_DOUBLE, 0, parent);
    }
    
    if(header_in[HEADER_BOOLEAN_COUNT] > 0) {
      MPI_Bcast(booleans_in, header_in[HEADER_BOOLEAN_COUNT], MPI_INTEGER, 0, parent);
    }
    
    if(header_in[HEADER_STRING_COUNT] > 0) {
      MPI_Bcast(string_sizes_in, header_in[HEADER_STRING_COUNT], MPI_INTEGER, 0, parent);
      
      int total_string_size = 0;
      for (int i = 0; i < header_in[HEADER_STRING_COUNT];i++) {
        total_string_size += string_sizes_in[i] + 1;
      }
      
      characters_in = new char[total_string_size];
      MPI_Bcast(characters_in, total_string_size, MPI_CHARACTER, 0, parent);

      int offset = 0;
      for (int i = 0 ; i <  header_in[HEADER_STRING_COUNT];i++) {
          strings_in[i] = characters_in + offset;
          offset += string_sizes_in[i] + 1;
      } 
    }

    header_out[HEADER_FLAGS] = 0;
    header_out[HEADER_CALL_ID] = header_in[HEADER_CALL_ID];
    header_out[HEADER_FUNCTION_ID] = header_in[HEADER_FUNCTION_ID];
    header_out[HEADER_CALL_COUNT] = call_count;
    header_out[HEADER_INTEGER_COUNT] = 0;
    header_out[HEADER_LONG_COUNT] = 0;
    header_out[HEADER_FLOAT_COUNT] = 0;
    header_out[HEADER_DOUBLE_COUNT] = 0;
    header_out[HEADER_BOOLEAN_COUNT] = 0;
    header_out[HEADER_STRING_COUNT] = 0;
    header_out[HEADER_UNITS_COUNT] = 0;

    //fprintf(stderr, "c worker mpi: handling call\n");
    
    must_run_loop = handle_call();
    
    //fprintf(stderr, "c worker mpi: call handled\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank == 0) {
      MPI_Send(header_out, HEADER_SIZE, MPI_INT, 0, 999, parent);
      
      if(header_out[HEADER_INTEGER_COUNT] > 0) {
        MPI_Send(ints_out, header_out[HEADER_INTEGER_COUNT], MPI_INT, 0, 999, parent);
      }
      if(header_out[HEADER_LONG_COUNT] > 0) {
        MPI_Send(longs_out, header_out[HEADER_LONG_COUNT], MPI_LONG_LONG_INT, 0, 999, parent);
      }
      if(header_out[HEADER_FLOAT_COUNT] > 0) {
        MPI_Send(floats_out, header_out[HEADER_FLOAT_COUNT], MPI_FLOAT, 0, 999, parent);
      }
      if(header_out[HEADER_DOUBLE_COUNT] > 0) {
        MPI_Send(doubles_out, header_out[HEADER_DOUBLE_COUNT], MPI_DOUBLE, 0, 999, parent);
      }
      if(header_out[HEADER_BOOLEAN_COUNT] > 0) {
        MPI_Send(booleans_out, header_out[HEADER_BOOLEAN_COUNT], MPI_INTEGER, 0, 999, parent);
      }
      if(header_out[HEADER_STRING_COUNT] > 0) {
        int offset = 0;
        for( int i = 0; i < header_out[HEADER_STRING_COUNT] ; i++) {
          
          int length = strlen(strings_out[i]);
          string_sizes_out[i] = length;
          offset += length + 1;
        }
        
        characters_out = new char[offset + 1];
        offset = 0;
        
        for( int i = 0; i < header_out[HEADER_STRING_COUNT]  ; i++) {
          strcpy(characters_out+offset, strings_out[i]);
          offset += string_sizes_out[i] + 1;
        }
        MPI_Send(string_sizes_out, header_out[HEADER_STRING_COUNT], MPI_INTEGER, 0, 999, parent);
        MPI_Send(characters_out, offset, MPI_BYTE, 0, 999, parent);
      }
    
    }
    
    if (characters_in) { 
        delete[] characters_in;
        characters_in = 0;
    }
    
    if (characters_out) {
        delete[] characters_out;
        characters_out = 0;
    }
    //fprintf(stderr, "call done\n");
  }
  delete_arrays();
  
    for(int i = 0; i < lastid + 1; i++) {
        MPI_Comm_disconnect(&communicators[i]);
    }
    
    MPI_Finalize();
  //fprintf(stderr, "mpi finalized\n");
#else
  fprintf(stderr, "mpi support not compiled into worker\n");
  exit(1);
#endif
}

void run_sockets_mpi(int argc, char *argv[], int port, char *host) {
#ifndef NOMPI
 bool must_run_loop = true;
  int max_call_count = 10;
  struct sockaddr_in serv_addr;
  struct hostent *server;
  int on = 1;
  int provided = 0;
  int rank = -1;
  
  mpiIntercom = false;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {  
    //fprintf(stderr, "C worker: running in sockets+mpi mode\n");
  
   
    socketfd = socket(AF_INET, SOCK_STREAM, 0);
    
    if (socketfd < 0) {
      perror("ERROR opening socket");
      //fprintf(stderr, "cannot open socket\n");
      exit(1);
    }

    //turn on no-delay option in tcp for huge speed improvement
    setsockopt (socketfd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof (on));
    
    server = gethostbyname(host);
    
    memset((char *) &serv_addr, '\0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy((char *) &serv_addr.sin_addr.s_addr, (char *) server->h_addr, server->h_length);
    serv_addr.sin_port = htons(port);
  
    if (connect(socketfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
      fprintf(stderr, "cannot connect socket to host %s, port %d\n", host, port);
      fprintf(stderr, "resolved IP address: %s\n",  inet_ntoa( * (struct in_addr *) server->h_addr));

      perror("ERROR connecting socket");
      //fprintf(stderr, "cannot connect socket\n");
      exit(1);
    }
    
    //fprintf(stderr, "sockets_mpi: finished initializing code\n");
  
    atexit(onexit_sockets);
  
  }
  
  header_in = new int[HEADER_SIZE];
  header_out = new int[HEADER_SIZE];

  new_arrays(max_call_count);  
  
  while(must_run_loop) {
    //fprintf(stderr, "sockets_mpi: receiving header\n");
    receive_array_sockets(header_in, HEADER_SIZE * sizeof(int), socketfd, rank);
    MPI_Bcast(header_in, HEADER_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    
    //fprintf(stderr, "C sockets_mpi worker code: got header %d %d %d %d %d %d %d %d %d %d\n", header_in[0], header_in[1], header_in[2], header_in[3], header_in[4], header_in[5], header_in[6], header_in[7], header_in[8], header_in[9]);
    
    int call_count = header_in[HEADER_CALL_COUNT];

    if (call_count > max_call_count) {
      delete_arrays();
      max_call_count = call_count + 255;
      new_arrays(max_call_count);
    }
    
    if (header_in[HEADER_INTEGER_COUNT] > 0) {
      receive_array_sockets(ints_in, header_in[HEADER_INTEGER_COUNT] * sizeof(int), socketfd, rank);
      MPI_Bcast(ints_in, header_in[HEADER_INTEGER_COUNT], MPI_INTEGER, 0, MPI_COMM_WORLD);
    }
     
    if (header_in[HEADER_LONG_COUNT] > 0) {
      receive_array_sockets(longs_in, header_in[HEADER_LONG_COUNT] * sizeof(long long int), socketfd, rank);
      MPI_Bcast(longs_in, header_in[HEADER_LONG_COUNT], MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    }
    
    if(header_in[HEADER_FLOAT_COUNT] > 0) {
      receive_array_sockets(floats_in, header_in[HEADER_FLOAT_COUNT] * sizeof(float), socketfd, rank);
      MPI_Bcast(floats_in, header_in[HEADER_FLOAT_COUNT], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    if(header_in[HEADER_DOUBLE_COUNT] > 0) {
      receive_array_sockets(doubles_in, header_in[HEADER_DOUBLE_COUNT] * sizeof(double), socketfd, rank);
      MPI_Bcast(doubles_in, header_in[HEADER_DOUBLE_COUNT], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    if(header_in[HEADER_BOOLEAN_COUNT] > 0) {
      //receive_array_sockets(booleans_in, header_in[HEADER_BOOLEAN_COUNT], socketfd , rank);
      for (int i = 0; i < header_in[HEADER_BOOLEAN_COUNT]; i++) {
        booleans_in[i] = 0;
        receive_array_sockets(&booleans_in[i], 1, socketfd , rank);
      }
      MPI_Bcast(booleans_in, header_in[HEADER_BOOLEAN_COUNT], MPI_INTEGER, 0, MPI_COMM_WORLD);
    }
    
    if(header_in[HEADER_STRING_COUNT] > 0) {
      receive_array_sockets(string_sizes_in, header_in[HEADER_STRING_COUNT] * sizeof(int), socketfd, rank);
      MPI_Bcast(string_sizes_in, header_in[HEADER_STRING_COUNT], MPI_INT, 0, MPI_COMM_WORLD);
      for (int i = 0; i < header_in[HEADER_STRING_COUNT]; i++) {
        strings_in[i] = new char[string_sizes_in[i] + 1];
        receive_array_sockets(strings_in[i], string_sizes_in[i], socketfd, rank);
        MPI_Bcast(strings_in[i], string_sizes_in[i], MPI_CHARACTER, 0, MPI_COMM_WORLD);
        strings_in[i][string_sizes_in[i]] = '\0';
      }
    }
    
    header_out[HEADER_FLAGS] = 0;
    header_out[HEADER_CALL_ID] = header_in[HEADER_CALL_ID];
    header_out[HEADER_FUNCTION_ID] = header_in[HEADER_FUNCTION_ID];
    header_out[HEADER_CALL_COUNT] = call_count;
    header_out[HEADER_INTEGER_COUNT] = 0;
    header_out[HEADER_LONG_COUNT] = 0;
    header_out[HEADER_FLOAT_COUNT] = 0;
    header_out[HEADER_DOUBLE_COUNT] = 0;
    header_out[HEADER_BOOLEAN_COUNT] = 0;
    header_out[HEADER_STRING_COUNT] = 0;
    header_out[HEADER_UNITS_COUNT] = 0;

    //fprintf(stderr, "c worker sockets_mpi: handling call\n");
    
    must_run_loop = handle_call();
    
    //fprintf(stderr, "c worker sockets_mpi: call handled\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {

      send_array_sockets(header_out, HEADER_SIZE * sizeof(int), socketfd, 0);
          
      if(header_out[HEADER_INTEGER_COUNT] > 0) {
        send_array_sockets(ints_out, header_out[HEADER_INTEGER_COUNT] * sizeof(int), socketfd, 0);
      }
          
      if(header_out[HEADER_LONG_COUNT] > 0) {
        send_array_sockets(longs_out, header_out[HEADER_LONG_COUNT] * sizeof(long long int), socketfd, 0);
      }
          
      if(header_out[HEADER_FLOAT_COUNT] > 0) {
        send_array_sockets(floats_out, header_out[HEADER_FLOAT_COUNT] * sizeof(float), socketfd, 0);
      }
          
      if(header_out[HEADER_DOUBLE_COUNT] > 0) {
        send_array_sockets(doubles_out, header_out[HEADER_DOUBLE_COUNT] * sizeof(double), socketfd, 0);
      }
          
      if(header_out[HEADER_BOOLEAN_COUNT] > 0) {
          for (int i = 0; i < header_out[HEADER_BOOLEAN_COUNT]; i++) {
            if (booleans_out[i]) {
              send_array_sockets(&TRUE_BYTE, 1, socketfd, 0);
            } else {
              send_array_sockets(&FALSE_BYTE, 1, socketfd, 0);
            }
         }
      }
          
      if(header_out[HEADER_STRING_COUNT] > 0) {
        for (int i = 0; i < header_out[HEADER_STRING_COUNT]; i++) {
          string_sizes_out[i] = strlen(strings_out[i]);
        }
        send_array_sockets(string_sizes_out, header_out[HEADER_STRING_COUNT] * sizeof(int), socketfd, 0);
          
        for (int i = 0; i < header_out[HEADER_STRING_COUNT]; i++) {
          send_array_sockets(strings_out[i], string_sizes_out[i] * sizeof(char), socketfd, 0);
        }
      }
        
      //fprintf(stderr, "sockets_mpicall done\n");
    }

  }
  delete_arrays();
  
  if (rank == 0) {
  
#ifdef WIN32
	closesocket(socketfd);
#else
	close(socketfd);
#endif
  }
  
  MPI_Finalize();
  
  //fprintf(stderr, "sockets_mpi done\n");
#else
  fprintf(stderr, "mpi support not compiled into worker\n");
  exit(1);
#endif
}

void run_sockets(int port, char *host) {
  bool must_run_loop = true;
  int max_call_count = 10;
  struct sockaddr_in serv_addr;
  struct hostent *server;
  int on = 1;

#ifdef WIN32
	WSADATA wsaData;
	int iResult;

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
	if (iResult != 0) {
	printf("WSAStartup failed: %d\n", iResult);
	exit(1);
	}
#endif
  mpiIntercom = false;

  //fprintf(stderr, "C worker: running in sockets mode\n");
   
  socketfd = socket(AF_INET, SOCK_STREAM, 0);
    
  if (socketfd < 0) {
    fprintf(stderr, "cannot open socket\n");
    exit(1);
  }
  
  //turn on no-delay option in tcp for huge speed improvement
  setsockopt (socketfd, IPPROTO_TCP, TCP_NODELAY, (const char *)&on, sizeof (on));
    
  server = gethostbyname(host);
    
  memset((char *) &serv_addr, '\0', sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  memcpy((char *) &serv_addr.sin_addr.s_addr, (char *) server->h_addr, server->h_length);
  serv_addr.sin_port = htons(port);
  
  if (connect(socketfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    fprintf(stderr, "cannot connect socket to host %s, port %d\n", host, port);
    fprintf(stderr, "resolved IP address: %s\n",  inet_ntoa( * (struct in_addr *) server->h_addr));

    perror("ERROR connecting socket");
    //fprintf(stderr, "cannot connect socket\n");
    exit(1);
  }
    
  //fprintf(stderr, "sockets: finished initializing code\n");
  
  atexit(onexit_sockets);
  
  header_in = new int[HEADER_SIZE];
  header_out = new int[HEADER_SIZE];

  new_arrays(max_call_count);  
  
  while(must_run_loop) {
    //fprintf(stderr, "sockets: receiving header\n");
    receive_array_sockets(header_in, HEADER_SIZE * sizeof(int), socketfd, 0);
    //fprintf(stderr, "C sockets worker code: got header %d %d %d %d %d %d %d %d %d %d\n", header_in[0], header_in[1], header_in[2], header_in[3], header_in[4], header_in[5], header_in[6], header_in[7], header_in[8], header_in[9]);
    
    int call_count = header_in[HEADER_CALL_COUNT];

    if (call_count > max_call_count) {
      delete_arrays();
      max_call_count = call_count + 255;
      new_arrays(max_call_count);
    }
    
    if (header_in[HEADER_INTEGER_COUNT] > 0) {
      receive_array_sockets(ints_in, header_in[HEADER_INTEGER_COUNT] * sizeof(int), socketfd, 0);
    }
     
    if (header_in[HEADER_LONG_COUNT] > 0) {
      receive_array_sockets(longs_in, header_in[HEADER_LONG_COUNT] * sizeof(long long int), socketfd, 0);
    }
    
    if(header_in[HEADER_FLOAT_COUNT] > 0) {
      receive_array_sockets(floats_in, header_in[HEADER_FLOAT_COUNT] * sizeof(float), socketfd, 0);
    }
    
    if(header_in[HEADER_DOUBLE_COUNT] > 0) {
      receive_array_sockets(doubles_in, header_in[HEADER_DOUBLE_COUNT] * sizeof(double), socketfd, 0);
    }
    
    if(header_in[HEADER_BOOLEAN_COUNT] > 0) {
      //receive_array_sockets(booleans_in, header_in[HEADER_BOOLEAN_COUNT], socketfd , 0);
      for (int i = 0; i < header_in[HEADER_BOOLEAN_COUNT]; i++) {
        booleans_in[i] = 0;
        receive_array_sockets(&booleans_in[i], 1, socketfd , 0);
      }
    }
    
    if(header_in[HEADER_STRING_COUNT] > 0) {
      receive_array_sockets(string_sizes_in, header_in[HEADER_STRING_COUNT] * sizeof(int), socketfd, 0);
      for (int i = 0; i < header_in[HEADER_STRING_COUNT]; i++) {
        strings_in[i] = new char[string_sizes_in[i] + 1];
        receive_array_sockets(strings_in[i], string_sizes_in[i], socketfd, 0);
        strings_in[i][string_sizes_in[i]] = '\0';
      }
    }
    
    header_out[HEADER_FLAGS] = 0;
    header_out[HEADER_CALL_ID] = header_in[HEADER_CALL_ID];
    header_out[HEADER_FUNCTION_ID] = header_in[HEADER_FUNCTION_ID];
    header_out[HEADER_CALL_COUNT] = call_count;
    header_out[HEADER_INTEGER_COUNT] = 0;
    header_out[HEADER_LONG_COUNT] = 0;
    header_out[HEADER_FLOAT_COUNT] = 0;
    header_out[HEADER_DOUBLE_COUNT] = 0;
    header_out[HEADER_BOOLEAN_COUNT] = 0;
    header_out[HEADER_STRING_COUNT] = 0;
    header_out[HEADER_UNITS_COUNT] = 0;


    //fprintf(stderr, "c worker sockets: handling call\n");
    
    must_run_loop = handle_call();
    
    //fprintf(stderr, "c worker sockets: call handled\n");

    send_array_sockets(header_out, HEADER_SIZE * sizeof(int), socketfd, 0);
      
    if(header_out[HEADER_INTEGER_COUNT] > 0) {
      send_array_sockets(ints_out, header_out[HEADER_INTEGER_COUNT] * sizeof(int), socketfd, 0);
    }
      
    if(header_out[HEADER_LONG_COUNT] > 0) {
      send_array_sockets(longs_out, header_out[HEADER_LONG_COUNT] * sizeof(long long int), socketfd, 0);
    }
      
    if(header_out[HEADER_FLOAT_COUNT] > 0) {
      send_array_sockets(floats_out, header_out[HEADER_FLOAT_COUNT] * sizeof(float), socketfd, 0);
    }
      
    if(header_out[HEADER_DOUBLE_COUNT] > 0) {
      send_array_sockets(doubles_out, header_out[HEADER_DOUBLE_COUNT] * sizeof(double), socketfd, 0);
    }
      
    if(header_out[HEADER_BOOLEAN_COUNT] > 0) {
        for (int i = 0; i < header_out[HEADER_BOOLEAN_COUNT]; i++) {
          if (booleans_out[i]) {
            send_array_sockets(&TRUE_BYTE, 1, socketfd, 0);
          } else {
            send_array_sockets(&FALSE_BYTE, 1, socketfd, 0);
          }
       }
    }
      
    if(header_out[HEADER_STRING_COUNT] > 0) {
      for (int i = 0; i < header_out[HEADER_STRING_COUNT]; i++) {
        string_sizes_out[i] = strlen(strings_out[i]);
      }
      send_array_sockets(string_sizes_out, header_out[HEADER_STRING_COUNT] * sizeof(int), socketfd, 0);
      
      for (int i = 0; i < header_out[HEADER_STRING_COUNT]; i++) {
        send_array_sockets(strings_out[i], string_sizes_out[i] * sizeof(char), socketfd, 0);
      }
    }
    
    //fprintf(stderr, "call done\n");
  }
  delete_arrays();
  
#ifdef WIN32
	closesocket(socketfd);
#else
	close(socketfd);
#endif
  //fprintf(stderr, "sockets done\n");
}
 
int main(int argc, char *argv[]) {
  int port;
  bool use_mpi;
  char *host;
  
  //for(int i = 0 ; i < argc; i++) {
  //  fprintf(stderr, "argument %d is %s\n", i, argv[i]);
  //}

  if (argc == 1) {
    run_mpi(argc, argv);
  } else if (argc == 4) {
    port = atoi(argv[1]);
    host = argv[2];
    
    if (strcmp(argv[3], "true") == 0) {
      use_mpi = true;
    } else if (strcmp(argv[3], "false") == 0) {
      use_mpi = false;
    } else {
      fprintf(stderr, "mpi enabled setting must be either 'true' or 'false', not %s\n", argv[2]);
      fprintf(stderr, "usage: %s [PORT HOST MPI_ENABLED]\n", argv[0]);
      exit(1);
    }    
    
    if (use_mpi) {
      run_sockets_mpi(argc, argv, port, host);
    } else {
      run_sockets(port, host);
    }
  } else {
    fprintf(stderr, "%s need either 0 or 4 arguments, not %d\n", argv[0], argc);
    fprintf(stderr, "usage: %s [PORT HOST MPI_ENABLED]\n", argv[0]);
    exit(1);
  }

  return 0;
}   


