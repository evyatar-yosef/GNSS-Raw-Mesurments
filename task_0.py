
import sys, os, csv
parent_directory = os.path.split(os.getcwd())[0]
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import navpy
from gnssutils import EphemerisManager
import simplekml
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8

def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100*np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3: # Stopping criterion
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1) # Distance between satellite and receiver
        # Eq. (1):
        phat = r + b0 # Predicted pseudorange
        # Eq. (3):
        deltaP = measured_pseudorange - phat # Pseudorange residual
        G[:, 0:3] = -(xs - x0) / r[:, None] # Partial derivative with respect to x
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP # Least squares solution
        # Eq. (5):
        dx = sol[0:3] # Update position
        db = sol[3] # Update clock bias
        x0 = x0 + dx # Update position
        b0 = b0 + db # Update clock bias
    norm_dp = np.linalg.norm(deltaP) # Norm of the pseudorange residual
    return x0, b0, norm_dp # Return the position, clock bias, and pseudorange residual

def parse_and_format_file(file_path):
    # Parse GNSS data from the log file into pandas DataFrames (tables)
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':  # Skip comments
                if 'Fix' in row[0]:  # Check if the row contains the header for the fix data
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]: # Check if the row contains the header for the raw data
                    measurements = [row[1:]]
            else: # If the row is not a comment
                if row[0] == 'Fix': # Check if the row contains fix data
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw': 
                    measurements.append(row[1:])
    # Create DataFrames from the parsed data
    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])
    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid'] # Add leading zero to single-digit IDs
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'# Assign GPS satellites
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'# Assign GLONASS satellites
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid'] # Combine constellation and ID to create satellite name

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G'] # Keep only GPS satellites

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz']) # Signal strength
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos']) # GNSS time
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos']) # Bias
    measurements['ReceivedSvTimeNanos']  = pd.to_numeric(measurements['ReceivedSvTimeNanos']) # Received satellite time
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond']) # Pseudorange rate
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos']) # Time uncertainty


    
    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' not in measurements.columns: # Check if the column exists
        measurements['BiasNanos'] = 0 # If not, create it and initialize it with zeros
    else:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos']) # If it exists, convert it to numeric

    if 'TimeOffsetNanos' not in measurements.columns: # Check if the column exists
        measurements['TimeOffsetNanos'] = 0 # If not, create it and initialize it with zeros
    else:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos']) # If it exists, convert it to numeric

    # Convert GNSS time to Unix time
    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0) # GPS epoch time
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch) # Convert to Unix time

    measurements['Epoch'] = 0 # Initialize epoch column
    measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum() # Assign epoch numbers

    # This should account for rollovers since it uses a week number specific to each measurement
    measurements['tRxGnssNanos'] = (measurements['TimeNanos'] + measurements['TimeOffsetNanos'] -
                                    (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])) # GNSS time in nanoseconds
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC) # GPS week number
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber'] # GNSS time in seconds
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos']) # GNSS time in seconds
    # Calculate pseudorange in seconds
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']
    
    
    # Conver to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds'] # Pseudorange in meters
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos'] # Pseudorange uncertainty in meters

    return android_fixes, measurements # Return the DataFrames

# Calculate satellite position based on ephemeris data and transmit times
def calc_coordinates_ecef(measurements, manager, sv_position, timestamp, sats):#
    ecef_list = []
    for epoch in measurements['Epoch'].unique(): # Iterate over each epoch
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)] # Select measurements with pseudorange less than 0.1 seconds
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName') # Drop duplicate measurements and set satellite name as index
        if len(one_epoch.index) > 4:# Check if there are at least 5 satellites
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)# Get timestamp for the epoch
            sats = one_epoch.index.unique().tolist() # Get satellite names
            ephemeris = manager.get_ephemeris(timestamp, sats) # Get ephemeris data for the satellites
            sv_position = calculate_satellite_position(ephemeris, one_epoch) # Calculate satellite positions

            xs = sv_position[['Sat_X', 'Sat_y', 'Sat_z']].to_numpy()
            pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
            pr = pr.to_numpy()

            a,b,dp = 0,0,0
            a, b, dp = least_squares(xs, pr, a, b)
            ecef_list.append(a)

    return ecef_list


def plot_position_offset(ned_df, show):
    plt.style.use('dark_background')
    plt.plot(ned_df['E'], ned_df['N'])
    plt.title('Position Offset From First Epoch')
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.gca().set_aspect('equal', adjustable='box')
    if (show):
        plt.show()
    else:
        pass


def run_navigation_analysis(file_path, ephemeris_data_directory):
    manager = EphemerisManager(ephemeris_data_directory)  # Use the passed directory path for EphemerisManager
    epoch = 0 # Initialize epoch counter
    num_sats = 0 # Initialize number of satellites

    android_fixes, measurements = parse_and_format_file(file_path) # Parse and format the file
    while num_sats < 5:  # Find epoch with at least 5 satellites
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)] # Select measurements with pseudorange less than 0.1 seconds
        one_epoch = one_epoch.drop_duplicates(subset='SvName') # Drop duplicate measurements
        if not one_epoch.empty: # Check if the epoch is not empty
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False) # Get timestamp for the epoch
            one_epoch.set_index('SvName', inplace=True) # Set satellite name as index
            num_sats = len(one_epoch.index) # Get number of satellites
        epoch += 1

    if num_sats >= 5:
        sats = one_epoch.index.unique().tolist() # Get satellite names
        ephemeris = manager.get_ephemeris(timestamp, sats) # Get ephemeris data for the satellites
        sv_position = calculate_satellite_position(ephemeris, one_epoch) # Calculate satellite positions
        sv_position.to_csv("satellite_coordinates.csv", sep=',') # Export satellite positions to CSV

        ecef_list = calc_coordinates_ecef(measurements, manager, sv_position, timestamp, sats) # Calculate ECEF coordinates

          

    # Perform coordinate transformations using the Navpy library
    ecef_array = np.stack(ecef_list, axis=0) # Convert list of ECEF coordinates to a NumPy array
    lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1) # Convert ECEF to LLA
    
    # Extract the first position as a reference for the NED transformation
    ref_lla = lla_array[0, :] # Reference LLA coordinates
    ned_array = navpy.ecef2ned(ecef_array, ref_lla[0], ref_lla[1], ref_lla[2]) # Convert ECEF to NED

    ned_df = pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

    # Save LLA and NED data to CSV files
    pd.DataFrame(lla_array, columns=['Latitude', 'Longitude', 'Altitude']).to_csv('lla_position.csv', index=False)
    pd.DataFrame(ned_array, columns=['N', 'E', 'D'])

    # Create DataFrame with Pos.X, Pos.Y, Pos.Z, Lat, Lon, Alt
    data_list = []

    # Iterate over the index range of the arrays (assuming both arrays have the same length)
    for i in range(len(ecef_array)):
        # Create a dictionary for each row with the corresponding data from ecef_array and lla_array
        row_data = { 
            'Pos.X': ecef_array[i, 0],
            'Pos.Y': ecef_array[i, 1],
            'Pos.Z': ecef_array[i, 2],
            'Latitude': lla_array[i, 0],
            'Longitude': lla_array[i, 1],
            'Altitude': lla_array[i, 2]
        } 
        # Append the dictionary to the list
        data_list.append(row_data)

    # Convert the list of dictionaries to a DataFrame
    data_df = pd.DataFrame(data_list)
    # Export the final DataFrame to CSV
    
    
    data_df.to_csv('calculated_postion.csv', index=False)
    # Export android fixes DataFrame to CSV
    android_fixes.to_csv('android_position.csv', index=False)
   
    # Notice: You need to download 'Geo Data Viewer' plugin to view the KML file
    plot_position_offset(ned_df, False) # Change to True to show the plot 

    # Save KML file
    new_file_path = file_path[:-4] + "-KML.kml" # Remove .txt and add -KML.kml
    kml = simplekml.Kml() # Initialize KML object
    index = 0
    while index < len(data_df): # Iterate over the rows of the DataFrame
        row = data_df.iloc[index] # Get the row at the current index
        kml.newpoint(name=str(index), coords=[(row['Longitude'], row['Latitude'], row['Altitude'])]) # Add a point to the KML file
        index += 1 # Increment the index
    kml.save(new_file_path) # Save the KML file
    print(f"Successfully generated {new_file_path} file.") # Print success message


# Calculate satellite position based on ephemeris data and transmit times
def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14 # Gravitational constant for Earth
    OmegaDot_e = 7.2921151467e-5 # Earth's rotation rate
    F = -4.442807633e-10 # Relativity correction coefficient
    sv_position = pd.DataFrame() # Initialize DataFrame for satellite positions
    sv_position['SatPRN'] = ephemeris.index # Satellite PRN
    sv_position.set_index('SatPRN', inplace=True)  # Set satellite PRN as index
    sv_position['GPS_time'] = transmit_time['tTxSeconds'] - ephemeris['t_oe'] # Time from ephemeris reference epoch
    A = ephemeris['sqrtA'].pow(2) # Semi-major axis
    n_0 = np.sqrt(mu / A.pow(3)) # Computed mean motion
    n = n_0 + ephemeris['deltaN'] # Corrected mean motion
    M_k = ephemeris['M_0'] + n * sv_position['GPS_time'] # Mean anomaly
    E_k = M_k # Eccentric anomaly
    err = pd.Series(data=[1] * len(sv_position.index)) # Initialize error
    i = 0
    # Calculate eccentric anomaly E_k using Newton's method
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k) # New values for E_k
        err = new_vals - E_k 
        E_k = new_vals # Update E_k
        i += 1

    sinE_k = np.sin(E_k) # Sine of E_k
    cosE_k = np.cos(E_k)    # Cosine of E_k
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k # Relativity correction
    delT_oc = transmit_time['tTxSeconds'] - ephemeris['t_oc']   # Time from clock reference epoch
    sv_position['delT_sv'] = (ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc +
                              ephemeris['SVclockDriftRate'] * delT_oc.pow(2)) 

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e'])) # True anomaly
    Phi_k = v_k + ephemeris['omega'] # Argument of latitude

    sin2Phi_k = np.sin(2 * Phi_k) # Sine of 2 * Phi_k
    cos2Phi_k = np.cos(2 * Phi_k) # Cosine of 2 * Phi_k
    
    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k 
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k
    
    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k # Radius
    
    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['GPS_time'] # Inclination

    x_k_prime = r_k * np.cos(u_k) # Satellite position in orbital plane
    y_k_prime = r_k * np.sin(u_k) # Satellite position in orbital plane
    
    Omega_k = (ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['GPS_time'] -
               OmegaDot_e * ephemeris['t_oe']) # Right ascension of ascending node

    # Calculate cos and sin only once for efficiency
    cos_Omega_k = np.cos(Omega_k)
    sin_Omega_k = np.sin(Omega_k)
    cos_i_k = np.cos(i_k)
    sin_i_k = np.sin(i_k)

    # Using numpy operations to calculate the satellite positions
    Sat_X = x_k_prime * cos_Omega_k - y_k_prime * cos_i_k * sin_Omega_k
    Sat_Y = x_k_prime * sin_Omega_k + y_k_prime * cos_i_k * cos_Omega_k
    Sat_Z = y_k_prime * sin_i_k

    # Assigning the results back to the sv_position DataFrame
    sv_position['Sat_X'] = Sat_X
    sv_position['Sat_y'] = Sat_Y
    sv_position['Sat_z'] = Sat_Z
    sv_position['CN0'] = transmit_time['Cn0DbHz'] # Signal strength

    return sv_position


if __name__ == "__main__":
        # Initialize paths outside of function
    parent_directory = os.path.split(os.getcwd())[0]
    ephemeris_data_directory = os.path.join(parent_directory, 'data')
    run_navigation_analysis('drive.txt', ephemeris_data_directory)
    run_navigation_analysis('stand.txt', ephemeris_data_directory)
    run_navigation_analysis('walk.txt', ephemeris_data_directory)
