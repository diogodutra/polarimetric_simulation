import copy
import numpy as np
import cmath
import math
import itertools
import datetime
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt

import time
import sys


light_speed = 3e8 # meters / s
_2pi = 2 * np.pi

    
default = {
    'frequency':           2.45e9,
    'position':            [0, 0, 0],
    'normal':              [1, 0, 0],
    'level_angle':         0,
    'power':               1,
    'phase':               0,
    'gain':                None,
    'name':                None, # receiver name
    'maximum_bounces':     9, # maximum quantity of bounces before quitting the loop
    'receiver':            False, # bool whether it is a facet or a receiver
    'verbose':             False, # print every step of every waves travelling
    'facet_length':        1e-1,
    'profiler':            False, # print calls and timelapse of some critical methods
    'cumulative_distance': 0,
    'cumulative_loss':     1,
}

user_parameters = {}


        

def modulus(vector):
    """Calculates the modulus of the vector.
    
    Args:
        vector (numpy.array): components of the vector.
        
    Returns:
        (float): norm (magnitude) of the vector.
    """
    return np.linalg.norm(vector)


def unit(vector, vector_modulus=None):
    """Calculates the unit vector.
    
    Args:
        vector (numpy.array): components of the vector.
        modulus (Optional float): magnitude of the vector for optimization (Default None).
        
    Returns:
        (numpy.array): components of the unit vector.
    """
    if vector_modulus is None: vector_modulus = modulus(np.array(vector))
    
    if vector_modulus<=0:
        warnings.warn("non-positive vector magnitude results in undetermined unit.")
    
    return np.array(vector) / vector_modulus
    

def decibels(power_ratio):
    """Calculates the decibels for the power ratio (dbW).
    
    Args:
        power_ratio (float): received power over the transmitted power.
        
    Returns:
        (numpy.array): power ratio in decibels [dbW].
    """
    return 10 * np.log10(power_ratio)


def degrees(radians):
    """Converts from radians to degrees.
    
    Args:
        radians (float): angle [radians].
        
    Returns:
        (float): angle [degrees].
    """
    return 180 / np.pi * radians


def cos_to_angle(cos):
    return math.acos(cos)


def angle_units(vector_1, vector_2):
    return math.acos(cossine_units(vector_1, vector_2))


def cossine_units(vector_1, vector_2):
    return np.clip(np.dot(vector_1, vector_2), -1.0, 1.0)
    

def angle(vector_1, vector_2):
    """Calculates the angle between two vectors.
    
    Args:
        vector_1 (numpy.array): components of the first vector.
        vector_2 (numpy.array): components of the second vector.
        
    Returns:
        (float): angle between the vectors [radians].
    """
    return angle_units(unit(vector_1), unit(vector_2))


def cis(angle=0):
    """Calculates the complex exponential function associated to an angle (aka Euler's formula).
    
    Args:
        angle (Optional float): angle [radians].
        
    Returns:
        (complex): complex exponential.
    """
    return cmath.exp(1j * angle)


def sphere_area(radius):
    """Calculates the area of a sphere.
    
    Args:
        radius (float): radius of the sphere [m].
        
    Returns:
        (float): area of the sphere [m2].
    """
    return 2 * _2pi * radius**2


def wavelength(frequency):
    """Calculates the wavelength of an electromagetic wave for a given frequency.
    
    Args:
        frequency (float): frequency of the electromagnetic wave [Hz].
        
    Returns:
        (float): wavelength of the electromagnetic wave [m].
    """
    return light_speed / frequency


def gain_antenna(*, peak=1, lobes=1, decay=.1):
    return lambda angle: max(0, math.exp(- decay * angle**2) * peak * math.cos(lobes * abs(angle)))

    
def gain_omni(peak=1):
    return gain_antenna(peak=peak, lobes=0, decay=0)


def gain_facet_mirror(*, peak=1, lobes=0, decay=10):
    return gain_antenna(peak=peak, lobes=lobes, decay=decay)


def gain_facet_rough(*, peak=.5, lobes=0, decay=.7):
    return gain_antenna(peak=peak, lobes=lobes, decay=decay)


def phase_shift(distance, wavelength, phase=0):
    """Calculates change in phase due to the wave travel.
    
    Args:
        distance (float): wave's path distance from origin to destination [m].
        wavelength (float): wavelength of the electromagnetic wave [m].
        phase (Optional float): wave's phase at the origin of the path [radians] (default 0).
        
    Returns:
        (float): wave's phase at the destination [radians].
    """
    phase_change = _2pi * distance * wavelength
    return (phase + phase_change) % _2pi


def power_loss(distance, power=1):
    """Free-space path loss due to lower power density of an isotropic radiator propagating without medium dispersion.
    
    Args:
        distance (float): wave's path distance from origin to destination [m].
        power (Optional float): wave's power at the origin of the path [W] (default 1).
        
    Returns:
        (float): wave's power at the destination [W].
    """
    density_loss = 1 / sphere_area(distance)
    return power * density_loss


def reflect_units(incident_unit, normal=[0, 0, 1], reflected_cos_angle=None):
    """Calculates the reflected vector given the incident and the surface's normal.
    
    Args:
        incident_unit (numpy.array): components of the unit incident vector.
        normal (numpy.array): components of the unit normal vector of the reflecting surface.
        reflected_cos_angle (Optional float): cossine of the angle of reflection, for optimization (Default None)
        
    Returns:
        (numpy.array): components of the unit reflected vector.
    """
    if reflected_cos_angle is None: reflected_cos_angle = cossine_units(incident_unit, normal)
    reflected_unit = incident_unit - 2 * reflected_cos_angle * normal
    return reflected_unit
    

class Wave():

    __slots__ = ['frequency',
                 'position',
                 'position_tuple',
                 'normal',
                 'power',
                 'phase',
                 'gain',
                 'wavelength',
                 'cumulative_distance',
                 'cumulative_loss',
                ] # optimization
        
    def __init__(self, frequency, position, normal, power, phase, gain, cumulative_distance=0, cumulative_loss=1):
        self.frequency = frequency
        self.wavelength = self.frequency / light_speed
        self.position = position
        self.position_tuple=tuple(position)
        self.normal = normal # assuming normal is unit
        self.power = power
        self.phase = phase
        self.gain = gain
        self.cumulative_distance = cumulative_distance
        self.cumulative_loss = cumulative_loss

        
    def __repr__(self):
        return f"<Wave" + \
               f" freq:{round(self.frequency)*1e-9}GHz" + \
               f" power:{round(decibels(self.power))}dbW" + \
               f" \u03C6:{round(degrees(self.phase))}\u00B0" + \
               f" position:{self.position}" + \
               f" normal:{self.normal}" + \
               f" distance:{round(self.cumulative_distance,1)}" + \
               f" loss:{self.cumulative_loss}" + \
               ">"
    
    
class Facet():

    __slots__ = ['area',
                 'position',
                 'position_tuple',
                 'normal',
                 'gain',
                 'receiver',
                 'name',
                ] # optimization
        
        
    def __init__(self, area, position, normal, gain, receiver, name):
        self.area = area
        self.position = position
        self.position_tuple=tuple(position)
        self.normal = normal # assuming normal is unit
        self.gain = gain
        self.receiver = receiver
        self.name = name

        
    def __repr__(self):
        return f"<Facet position:{self.position} normal:{self.normal}>"
        
        
class Simulation():
    """Contains all waves, facets, receivers and methods to run the polarimetric simulation.

        Args:
            verbose (Optional bool): prints every wave bounce (default False)

        Usage::
        
        >>> receiver_name = 'Receiver 1'
        >>> frequencies = np.arange(2.1, 2.9, .01)*1e9
        >>> distances = np.arange(1, 1.3, .1)
        >>> height = .4
        >>> width = 1.6
        
        >>> fig, axs = plt.subplots(2, 1)
        >>> for distance in distances:
        >>>     sim = simulation.Simulation()
        >>>     sim.add_plate(width=width, height=height, position=[distance, 0, 0], normal=[-1, 0, 0], facet_length=facet_length)

        >>>     sim.add_receiver(name=receiver_name)

        >>>     for frequency in frequencies:
        >>>         sim.add_wave(frequency=frequency, normal=[1, 0, 0])
        
        >>>     sim.run()
        >>>     sim.plot_receiver(receiver_name, axes=axs)   
    
        >>> _ = plt.legend(np.round(distances,1), title='distance')
        >>> plt.suptitle(f'Svv from plate {round(width,1)}m x {round(height,1)}m')
    """
    
    
    def __init__(self, **kwargs):
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)
        
        self.frequencies = set()
        self.measurements = {} # sum of all waves per receiver
        self.waves = []
        self.facets = [] # infinitesimal surfaces
        self.receivers = {}
        self.geometries = {} # geometry of each pair of facets
        self.valid_paths = defaultdict(lambda: []) # valid wave destinations        

        
    def __repr__(self):
        waves = len(self.waves)
        receivers = len(self.receivers.keys())
        facets = len(self.facets) - receivers
        return f"<Simulation waves:{waves} facets:{facets} receivers:{receivers}>"
    
    
    def _include_default_parameters(self, **kwargs):
        # copy default parameters
        user_parameters = copy.copy(default)
        
        # update parameters with user's kwargs
        user_parameters.update(kwargs)
        
        # convert list to np.array
        for key in ['position', 'normal']:
            user_parameters[key] = np.array(user_parameters[key])
        
        return user_parameters
    
    
    def add_wave(self, **kwargs):
        """Adds one electromagnetic wave to the simulation.

        Args:
            frequency (float): frequency of the electromagnetic wave [Hz].
            power (Optional float): wave's transmitted power [dbW] (Default 1).
            phase (optional float): wave's transmitted phase [radians] (Default 0).
            position (Optional numpy.array): components of position (Default [0, 0, 0]).
            normal (Optional numpy.array): components of normal (Default [1, 0, 0]).
            gain (Optional function): gain of the transmitter (Default lambda angle: max(0, .5*np.cos(5*abs(angle))**.5))
        """
        kwargs = self._include_default_parameters(**kwargs)

        if kwargs['gain'] is None: kwargs['gain'] = gain_omni()

        self.waves.append(Wave(
                frequency=kwargs['frequency'], # assuming no Doppler effect
                position=kwargs['position'],
                normal=kwargs['normal'],
                power=kwargs['power'],
                phase=kwargs['phase'],
                gain=kwargs['gain'],
                cumulative_distance=kwargs['cumulative_distance'],
                cumulative_loss=kwargs['cumulative_loss']
            ))
        
        self.frequencies.add(kwargs['frequency'])
        
        
    def add_facet(self, **kwargs):
        """Adds one facet (element of the discrete surface) to the simulation.

        Args:
            area (Optional float): area of the facet [m2] (Default 1e-2).
            gain (Optional lambda function): radiation pattern for the reflection (Default  gain_facet_rough)
            position (Optional numpy.array): components of the position (Default [0, 0, 0]).
            normal (Optional numpy.array): components of the normal to the surface (Default [0, 1, 0]).
        """
        kwargs = self._include_default_parameters(**kwargs)

        if kwargs['gain'] is None: kwargs['gain'] = gain_facet_rough()
            
        # add new Facet
        kwargs.update(receiver=False)
        self.facets.append(Facet(
            area=kwargs['facet_length']**2,
            position=kwargs['position'],
            normal=kwargs['normal'],
            gain=kwargs['gain'],
            receiver=False,
            name=None,
        ))
        
         
    def add_plate(self, width=5, height=.1, **kwargs):
        """Adds to the simulation a vertical plane as a series of facets normal=[-1, 0, 0].

        Args:
            width (Optional float): width of the square plate [m] (Default 5).
            height (Optional float): height of the square plate [m] (Default 0.1).
            area (Optional float): area of each facet [m2] (Default 1e-2).
            gain (Optional function): radiation pattern for the reflection of each facet (Default lambda angle: max(0, .5*math.cos(5*abs(angle))**.5))
            position (Optional numpy.array): components of the position (Default [0, 0, 0].
            normal (Optional numpy.array): components of the normal to the surface (Default [0, 1, 0]).
            
        TODO:
            - Add dimensions of the plate as argument.
            - Add position of each facet aligned to the normal.
        """
        kwargs = self._include_default_parameters(**kwargs)
        
        facets_width = np.arange(-(width-kwargs['facet_length'])/2, (width+kwargs['facet_length'])/2, kwargs['facet_length'])
        facets_height = np.arange(-(height-kwargs['facet_length'])/2, (height+kwargs['facet_length'])/2, kwargs['facet_length'])
        
        for y, z in itertools.product(facets_width, facets_height):
            position = kwargs['position'] + np.array([0, y, z])
            local_kwargs = copy.copy(kwargs)
            local_kwargs.update(position=position)            
            self.add_facet(**local_kwargs)
        
        
    def add_receiver(self, **kwargs):
        """Adds an omini-directional receiver to the simulation in order to measure the environment.

        Args:
            name (Optional string): receiver's label (Default current date-time).
            area (Optional float): area of each facet [m2] (Default 1e-2).
            position (Optional numpy.array): components of the position (Default [0, 0, 0].
            normal (Optional numpy.array): components of the normal to the surface of the receiver plane (Default [1, 0, 0]).
            
        TODO:
            - Add gain of the receiver's antenna as argument.
        """
        kwargs = self._include_default_parameters(**kwargs)
        
        # get name of the receiver
        name = kwargs['name']
        if name is None: name = str(datetime.datetime.now())
            
        if kwargs['gain'] is None: kwargs['gain'] = gain_antenna()
            
        # add one Facet at this new receiver position
        kwargs.update(name=name)
        kwargs.update(receiver=True)
        
        self.facets.append(Facet(
            area=kwargs['facet_length']**2,
            position=kwargs['position'],
            normal=kwargs['normal'],
            gain=kwargs['gain'],
            receiver=True,
            name=name,
        ))
        
        # add receiver to the dictionary member with empty list of received waves
        self.receivers[name] = []
        
        
    def _append_reflected_wave(self, new_wave, facet):
        is_receiver = facet.receiver
        if is_receiver:

            # TODO: add receiver gain

            # add new wave to the list of this received waves
            self.receivers[facet.name].append(new_wave)

        else:
            # add new wave to the list of reflections
            self.reflections.append(new_wave)
    
    
    def _convert_waves_to_measurements(self):            
        # convert list of waves to single measurement for every frequency
        self.frequencies = np.array(sorted(list(self.frequencies)))
        for receiver_name, receiver_waves in self.receivers.items():
            measurement_per_frequency = {frequency: 0+0j for frequency in self.frequencies}
            for wave in receiver_waves:
                power_complex = wave.power * wave.cumulative_loss * power_loss(wave.cumulative_distance) * cis(wave.phase + phase_shift(wave.cumulative_distance, wave.wavelength))
                measurement_per_frequency[wave.frequency] += power_complex
                
        self.measurements[receiver_name] = measurement_per_frequency


    def plot_receiver(self, receiver_name, axes=None):
        """Plots the signal from a receiver.

        Args:
            receiver_name (string): receiver's label.
        """
        
        if axes is None: fig, axes = plt.subplots(2, 1, sharex=True)

        power_phase = np.zeros((len(self.frequencies), 2))
        measurement_per_frequency = self.measurements[receiver_name]

        for i_freq, (frequency, power_complex) in enumerate(measurement_per_frequency.items()):
            power = np.abs(power_complex)
            phase = np.angle(power_complex) % _2pi

            power_phase[i_freq, 0] = power
            power_phase[i_freq, 1] = phase


        axes[0].plot(self.frequencies*1e-9, decibels(power_phase[:,0]))
        axes[0].set_ylabel('dbW')

        axes[1].plot(self.frequencies*1e-9, degrees(power_phase[:,1]))
        axes[1].set_xlabel('Frequency [GHz]')
        axes[1].set_ylabel('\u03C6 [\u00B0]')

        plt.subplots_adjust(hspace=0)
        _ = plt.suptitle(f'Rvv from {receiver_name}')
        
        
    def plot_gain(self, gain):
        """Plots the radiation pattern.

        Args:
            gain (lambda): function of angle in radians.
        """       
        angles = np.arange(-np.pi, np.pi, .001)
        plt.polar(angles, np.array(list(map(gain, angles))))
        plt.title('Radiation Pattern')
        plt.xlabel('Angle [degrees]')
        plt.ylabel('Gain')
        plt.savefig('Radiation_Pattern')
        
        
    def _run_step(self):
        """Runs one bounce for every current wave in the simulation."""
        
        if self.profiler: times = defaultdict(lambda: [0, 0])
        
        self.reflections = []
        
        paths_for_this_step = self._list_paths_for_this_step()
            
        for wave, facets in paths_for_this_step:
            for (facet, key) in facets:
                    
                if self.profiler: start_time = time.time()
                geometry = self.geometries[key]
                if self.profiler: times['geometry'][0] += 1
                if self.profiler: times['geometry'][1] += time.time() - start_time

                if self.verbose: print_text = f'Wave freq:{wave.frequency} @ {wave.position} dbW:{round(decibels(wave.power))} ' + \
                    f'\u03C6:{round(degrees(wave.phase))}\u00B0 going to {facet.position}'

                if self.profiler: start_time = time.time()
                gain = wave.gain(geometry['reflected_angle_1']) # assuming azimuth==elevation in radiation pattern
                if self.profiler: times['gain    '][0] += 1
                if self.profiler: times['gain    '][1] += time.time() - start_time

                if gain <= 0:

                    if self.verbose: print_text += ' but gone because gain is zero.'

                else:

                    if self.profiler: start_time = time.time()
                    new_wave = Wave(
                            frequency=wave.frequency, # assuming no Doppler effect
                            position=facet.position,
                            normal=geometry['reflected_unit_2'],
                            power=wave.power,
                            phase=wave.phase,
                            gain=facet.gain,
                            cumulative_distance= wave.cumulative_distance + geometry['distance'],
                            cumulative_loss= wave.cumulative_loss * gain * facet.area
                        )
                    if self.profiler: times['new_wave'][0] += 1
                    if self.profiler: times['new_wave'][1] += time.time() - start_time

                    if self.profiler: start_time = time.time()
                    self._append_reflected_wave(new_wave, facet)
                    if self.profiler: times['append  '][0] += 1
                    if self.profiler: times['append  '][1] += time.time() - start_time


                if self.verbose: print(print_text)
                  
        self.waves = self.reflections
        
        if self.profiler:
            for time_key, (time_calls, time_duration) in times.items():            
                print(f" ┌ {time_key}:\t {time_calls} calls,\t {time_duration} s")
                
                
    def _list_paths_for_this_step(self):
        valid_paths = []
        for wave in self.waves:
            valid_destinations = self.valid_paths[wave.position_tuple]
            if len(valid_destinations) > 0:
                valid_paths.append([wave, valid_destinations])
            
            
        return valid_paths
        
        
    def _geometry(self, facet_1, facet_2):
        
        geometry = {'valid': False,
                  'displacement': None,
                  'distance': None,
                  'direction': None,
                  'reflected_cos_1': None,
                  'reflected_cos_2': None,
                  'reflected_angle_1': None,
                  'reflected_angle_2': None,
                  'reflected_unit_1': None,
                  'reflected_unit_2': None,
                 }
        
        geometry['displacement'] = facet_1.position - facet_2.position
        geometry['distance'] = modulus(geometry['displacement'])

        is_valid = geometry['distance'] > 0 # ignore when both positions are identical
            
        if is_valid:
            
            geometry['direction'] = unit(geometry['displacement'], geometry['distance'])
            
            geometry['reflected_cos_1'] = -cossine_units(facet_1.normal,  geometry['direction'])
            geometry['reflected_cos_2'] = -cossine_units(facet_2.normal, -geometry['direction'])
            
            is_facing_1 = geometry['reflected_cos_1']
            is_facing_2 = geometry['reflected_cos_2']
            is_valid = is_facing_1 and is_facing_2
            
            if is_valid:  # BUG: omini antennas should not be discarded
                geometry['reflected_angle_1'] = cos_to_angle(geometry['reflected_cos_1'])
                geometry['reflected_angle_2'] = cos_to_angle(geometry['reflected_cos_2'])
                
                reflected_unit_1 = reflect_units(geometry['direction'], facet_1.normal, geometry['reflected_cos_1'])
                reflected_unit_2 = reflect_units(geometry['direction'], facet_2.normal, geometry['reflected_cos_2'])
                
                geometry['reflected_unit_1'] = reflected_unit_1
                geometry['reflected_unit_2'] = reflected_unit_2
                
                geometry['valid'] = True
                  
        
        return geometry
    
    
    def _invert_path(self, geometry):        
        return {  'valid': geometry['valid'],
                  'displacement': geometry['displacement'],
                  'distance': geometry['distance'],
                  'direction': geometry['direction'],
                  'reflected_cos_1': geometry['reflected_cos_2'],
                  'reflected_cos_2': geometry['reflected_cos_1'],
                  'reflected_angle_1': geometry['reflected_angle_2'],
                  'reflected_angle_2': geometry['reflected_angle_1'],
                  'reflected_unit_1': geometry['reflected_unit_2'],
                  'reflected_unit_2': geometry['reflected_unit_1'],
                 }
    
    
    def _calculate_paths(self):
            
        for facet_1, facet_2 in itertools.combinations(self.facets + self.waves, r=2):
            
            # calculate geometry of the path
            key = (facet_1.position_tuple, facet_2.position_tuple)
            if key not in self.geometries.keys():
                
                geometry = self._geometry(facet_1, facet_2)
                self.geometries[key] = geometry
                inverted_key = (facet_2.position_tuple, facet_1.position_tuple)
                self.geometries[inverted_key] = self._invert_path(geometry)
                
                
                # add this to the list of valid paths
                if geometry['valid']:
                    self.valid_paths[facet_1.position_tuple].append((facet_2, key))
                    self.valid_paths[facet_2.position_tuple].append((facet_1, inverted_key))
            
            
    def _preprocess(self):
        
        # error dealing ----------------------
        if len(self.waves) == 0:
            raise ValueError('No waves found. Add at least one with the "add_wave" method before running the simulation.')
        
        if len(self.receivers) == 0:
            raise ValueError('No receivers found. Add at least one with the "add_receiver" method before running the simulation.')
        
        if len(self.facets) - len(self.receivers) == 0:
            raise ValueError('No facets found. Add at least one with the "add_facet" method before running the simulation.')
        
        # preprocessing geometry -----------
        self._calculate_paths()
    
        
    def run(self, **kwargs):
        """Runs all bounces for every current wave in the simulation.

        Args:
            - maximum_bounces (Optional int): upper limit to interrupt simulation (Default 9).
        """
        
        if self.profiler: start_time = time.time()
        self._preprocess()
        if self.profiler: print("Preprocess: %s seconds" % (time.time() - start_time))
        
        kwargs = self._include_default_parameters(**kwargs)
        
        maximum_bounces = kwargs['maximum_bounces']
        
        for bounce in range(maximum_bounces):
            
            if self.verbose: print(f'Bounce {bounce} with {len(self.waves)} waves')
            
            if self.profiler: start_time = time.time()
            
            self._run_step()
    
            if self.profiler: print("Bounce: %s seconds" % (time.time() - start_time))

            stop_loop = (len(self.waves) <= 0)
            if stop_loop: break

        if bounce == maximum_bounces - 1:
            warnings.warn("simulation stopped earlier because it reached the maximum bounces.")
            
            
        if self.profiler: start_time = time.time()
        self._convert_waves_to_measurements()
        if self.profiler: print("Measurements: %s seconds" % (time.time() - start_time))