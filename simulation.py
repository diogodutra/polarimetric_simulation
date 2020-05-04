import copy
import numpy as np
import cmath
import math
import itertools
import datetime
import warnings


light_speed = 3e8 # meters / s
    
default = {
    'frequency':       2.45e9,
    'position':        [0, 0, 0],
    'normal':          [1, 0, 0],
    'area':            1e-2,
    'boresight':       [1, 0, 0],
    'level_angle':     0,
    'power':           1,
    'phase':           0,
    'power_cutoff':    1e-9, # (ie: 1e-9 = -90 decibels)
    'gain':            lambda incident_angle: max(0, .5*np.cos(5*abs(incident_angle))**.5),
    'name':            None, # receiver name
    'maximum_bounces': 9, # maximum quantity of bounces before quitting the loop
    'receiver':        False, # bool whether it is a facet or a receiver
    'verbose':         False, # print every step of every waves travelling
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


def unitary(vector):
    """Calculates the unitary vector.
    
    Args:
        vector (numpy.array): components of the vector.
        
    Returns:
        (numpy.array): components of the unitary vector.
    """
    vector_abs = modulus(np.array(vector))
    
    if vector_abs==0:
        warnings.warn("null vector input to unitary function resulted in null vector.")
    
    return np.array(vector) / vector_abs
    

def decibels(power_ratio):
    """Calculates the decibels for the power ratio (dbW).
    
    Args:
        power_ratio (float): received power over the transmitted power.
        
    Returns:
        (numpy.array): power ratio in decibels [dbW].
    """
    return 10 * np.log10(power_ratio)
    

def angle(vector_1, vector_2):
    """Calculates the angle between two vectors.
    
    Args:
        vector_1 (numpy.array): components of the first vector.
        vector_2 (numpy.array): components of the second vector.
        
    Returns:
        (float): angle between the vectors [radians].
    """
    return math.acos(np.clip(np.dot(unitary(vector_1), unitary(vector_2)), -1.0, 1.0))


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
    return 4 * np.pi * radius**2


def wavelength(frequency):
    """Calculates the wavelength of an electromagetic wave for a given frequency.
    
    Args:
        frequency (float): frequency of the electromagnetic wave [Hz].
        
    Returns:
        (float): wavelength of the electromagnetic wave [m].
    """
    return light_speed / frequency


def propagate(distance, frequency, power=1, phase=0):
    """Free-space path loss due to lower power density of an isotropic radiator propagating without medium dispersion.
    
    Args:
        distance (float): wave's path distance from origin to destination [m].
        frequency (float): frequency of the electromagnetic wave [Hz].
        power (Optional float): wave's power at the origin of the path [W] (default 1).
        phase (Optional float): wave's phase at the origin of the path [radians] (default 0).
        
    Returns:
        (float): wave's power at the destination [W].
        (float): wave's phase at the destination [radians].
    """
    density_loss = 1 / sphere_area(distance)
    phase_change = 2 * np.pi * distance * (frequency / light_speed)
    return power * density_loss, (phase + phase_change) % (2*np.pi)


def reflect(incident, normal=[0, 0, 1]):
    """Calculates the reflected vector given the incident and the surface's normal.
    
    Args:
        incident (numpy.array): components of the unitary incident vector.
        normal (numpy.array): components of the unitary normal vector of the reflecting surface.
        
    Returns:
        (numpy.array): components of the unitary reflected vector.
    """
    incident = unitary(incident)
    normal = unitary(normal)
    reflected_angle = angle(incident, normal)
    reflected_unitary = incident - 2 * np.dot(incident, normal) * normal
    return reflected_angle, reflected_unitary    
    

class Wave():
    
    def __init__(self, **kwargs):
#         self.position = np.array(kwargs['position'])
#         self.frequency = kwargs['frequency']
#         self.boresight = unitary(np.array(kwargs['boresight']))
#         self.gain = kwargs['gain']
#         self.power = kwargs['power']
#         self.phase = kwargs['phase']
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)

        
    def __repr__(self):
        return f"<Wave" + \
               f" freq:{round(self.frequency)*1e-9}GHz" + \
               f" power:{round(decibels(self.power))}dbW" + \
               f" \u03C6:{round(self.phase * 180 / np.pi)}\u00B0" + \
               f" position:{self.position}m>"
    
    
class Facet():
    
    def __init__(self, **kwargs):
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)
        
#         print(kwargs['normal'])
#         self.normal = unitary(np.array(kwargs['normal']))
#         self.position = np.array(kwargs['position'])
#         self.gain = kwargs['gain'] # gain_reflection
#         self.name = kwargs['name']
#         self.area = kwargs['area']
#         self.receiver = kwargs['receiver']

        
    def __repr__(self):
        return f"<Facet position:{self.position} normal:{self.normal}>"
        
        
class Simulation():
    """Contains all waves, facets, receivers and methods to run the polarimetric simulation.

        Args:
            verbose (Optional bool): prints every wave bounce (default False)

        Usage::
        
        >>> sim = simulation.Simulation()
        
        >>> sim.add_facet(position=[1, 0, 0], normal=[-1, 0, 0])        
        >>> sim.add_receiver(boresight=[1, 0, 0], name='Receiver 1')
        >>> sim.add_wave(frequency=2.45e9, boresight=[1, 0, 0])
        
        >>> sim.run()
    """
        
    waves = [] # list of active waves
    facets = [] # list of infinitesimal surfaces
    receivers = {} # dictionary of receivers
    
    
    def __init__(self, **kwargs):
        self.__dict__ = copy.copy(default)
        
        # update parameters with user kwargs
        self.__dict__.update(kwargs)
        
        self.frequencies = set()
        self.measurements = {}

        
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
        for key in ['position', 'boresight', 'normal']:
            user_parameters[key] = np.array(user_parameters[key])
        
        return user_parameters
    
    
    def add_wave(self, **kwargs):
        """Adds one electromagnetic wave to the simulation.

        Args:
            frequency (float): frequency of the electromagnetic wave [Hz].
            power (Optional float): wave's transmitted power [dbW] (Default 1).
            phase (optional float): wave's transmitted phase [radians] (Default 0).
            position (Optional numpy.array): components of position (Default [0, 0, 0].
            boresight (Optional numpy.array): components of boresight (Default [1, 0, 0].
            gain (Optional function): gain of the transmitter (Default lambda angle: max(0, .5*np.cos(5*abs(angle))**.5))
        """
        kwargs = self._include_default_parameters(**kwargs)
            
        # add new Wave
        self.waves.append(Wave(**kwargs))
        
        self.frequencies.add(kwargs['frequency'])
        
        
    def add_facet(self, **kwargs):
        """Adds one facet (element of the discrete surface) to the simulation.

        Args:
            area (Optional float): area of the facet [m2] (Default 1e-2).
            gain (Optional function): radiation pattern for the reflection (Default lambda angle: max(0, .5*np.cos(5*abs(angle))**.5))
            position (Optional numpy.array): components of the position (Default [0, 0, 0]).
            normal (Optional numpy.array): components of the normal to the surface (Default [0, 1, 0]).
        """
        kwargs = self._include_default_parameters(**kwargs)
            
        # add new Facet
        kwargs.update(receiver=False)
        self.facets.append(Facet(**kwargs))
        
         
    def add_square_plate(self, **kwargs):
        """Adds to the simulation a vertical plane as a series of facets normal=[-1, 0, 0].

        Args:
            area (Optional float): area of each facet [m2] (Default 1e-2).
            gain (Optional function): radiation pattern for the reflection of each facet (Default lambda angle: max(0, .5*np.cos(5*abs(angle))**.5))
            position (Optional numpy.array): components of the position (Default [0, 0, 0].
            normal (Optional numpy.array): components of the normal to the surface (Default [0, 1, 0]).
            
        TODO:
            - Add dimensions of the plate as argument.
            - Add position of each facet aligned to the normal.
        """
        kwargs = self._include_default_parameters(**kwargs)
        
        distance_facets = .1
        n_facets_each_side = 5
        
        list_facets = range(-n_facets_each_side, n_facets_each_side + 1)
        for x, y in itertools.product([0,], list_facets):
            position = kwargs['position'] + distance_facets * np.array([x, y, 0])
            local_kwargs = copy.copy(kwargs)
            local_kwargs.update(position=position)            
            self.add_facet(**local_kwargs)
        
        
    def add_receiver(self, **kwargs):
        """Adds an omini-directional receiver to the simulation in order to measure the environment.

        Args:
            name (Optional string): receiver's label (Default current date-time).
            area (Optional float): area of each facet [m2] (Default 1e-2).
            position (Optional numpy.array): components of the position (Default [0, 0, 0].
            normal (Optional numpy.array): components of the boresight or normal to the surface (Default [1, 0, 0]).
            
        TODO:
            - Add gain of the receiver's antenna as argument.
        """
        kwargs = self._include_default_parameters(**kwargs)
        
        # get name of the receiver
        name = kwargs['name']
        if name is None: name = str(datetime.datetime.now())
            
        # add one Facet at this new receiver position
        kwargs.update(name=name)
        kwargs.update(receiver=True)
        self.facets.append(Facet(**kwargs))
        
        # add receiver to the dictionary member with empty list of received waves
        self.receivers[name] = []
    
    
    def _convert_waves_to_measurements(self):            
        # convert list of waves to single measurement for every frequency
        self.frequencies = sorted(list(self.frequencies))
        for receiver_name, receiver_waves in self.receivers.items():
            measurement_per_frequency = {frequency: 0+0j for frequency in self.frequencies}
            for wave in receiver_waves:
                power_complex = wave.power * cis(wave.phase)
                measurement_per_frequency[wave.frequency] += power_complex
                
        self.measurements[receiver_name] = measurement_per_frequency
        
        
    def run_step(self):
        """Runs one bounce for every current wave in the simulation."""
        
        reflections = []
        
        for wave, facet in itertools.product(self.waves, self.facets):
            
            # geometry calculations
            displacement = facet.position - wave.position
            distance = modulus(displacement)
    
            if distance > 0: # ignore when facet and wave are at the exact same position

                reflected_angle, reflected_unitary = reflect(displacement, facet.normal)
                reflected_angle = abs(reflected_angle - np.pi)
                
                if self.verbose: print_text = f'Wave freq:{wave.frequency} @ {wave.position}m P={round(decibels(wave.power))}dbW ' + \
                    f'\u03C6={round(wave.phase*180/np.pi)}\u00B0 going to {facet.position}m' + \
                    f' with incidence={round(reflected_angle*180/np.pi)}\u00B0'
                
                is_behind = (np.cos(reflected_angle) <= 0)
                
                if is_behind:
                    
                    if self.verbose: print_text += ' but gone because it approached from behind.'
                    
                else:
                    
                    boresight_angle = angle(wave.boresight, displacement)
                    
                    gain = wave.gain(boresight_angle) # assuming azimuth==elevation in radiation pattern
                    
                    if gain <= 0:
                    
                        if self.verbose: print_text += ' but gone because gain is zero.'
                            
                    else:

                        # electromagnetic calculations
                        power_origin = wave.power * gain
                        power_incident, phase_incident = propagate(distance, wave.frequency, power_origin, wave.phase)
                        phase_incident = phase_incident % (2*np.pi)
                        power_incident *= facet.area # assuming power constant over all the facet surface. Is it really proportional to the facet area?

                        is_faded_out = power_incident < default['power_cutoff']

                        if is_faded_out:

                            if self.verbose: print_text += f' but gone because faded out with P={round(decibels(power_incident))}dbW.'

                        else:

                            new_wave = Wave(
                                    power=power_incident,
                                    phase=phase_incident,
                                    frequency=wave.frequency, # assuming no Doppler effect
                                    position=facet.position,
                                    gain=facet.gain,
                                    boresight=reflected_unitary,
                                )

                            is_receiver = facet.receiver
                            if is_receiver:

                                # TODO: add receiver gain

                                # add new wave to the list of this received waves
                                self.receivers[facet.name].append(new_wave)
                                
                            else:
                                # add new wave to the list of reflections
                                reflections.append(new_wave)
                                
                
                if self.verbose: print(print_text)
                     
        self.waves = reflections
        
        
    def run(self, **kwargs):
        """Runs all bounces for every current wave in the simulation.

        Args:
            - maximum_bounces (Optional int): upper limit to interrupt simulation (Default 9).
        """
        
        if len(self.waves) == 0:
            raise ValueError('No waves found. Add at least one with the "add_wave" method before running the simulation.')
        
        if len(self.receivers) == 0:
            raise ValueError('No receivers found. Add at least one with the "add_receiver" method before running the simulation.')
        
        if len(self.facets) - len(self.receivers) == 0:
            raise ValueError('No facets found. Add at least one with the "add_facet" method before running the simulation.')
        
        kwargs = self._include_default_parameters(**kwargs)
        
        maximum_bounces = kwargs['maximum_bounces']
        
        for bounce in range(maximum_bounces):
            
            if self.verbose: print(f'Bounce {bounce} with {len(self.waves)} waves')

            self.run_step()

            stop_loop = (len(self.waves) <= 0)
            if stop_loop:
                break

        if bounce == maximum_bounces - 1:
            warnings.warn("simulation stopped earlier because it reached the maximum bounces.")
            
            
        self._convert_waves_to_measurements()