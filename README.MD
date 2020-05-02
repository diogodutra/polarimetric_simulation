# Simulation of a polarimetric radar in a 3D scene

[![Generic badge](https://img.shields.io/badge/Python-3.6-blue.svg)](https://shields.io/)

### Example:
    sim = mimo.simulation.Simulation()
    
    receiver_name = 'Receiver 1'
    transmitter_power = 1
    
    gain_antenna = lambda incident_angle: max(0, 1 - abs(incident_angle))
    gain_receiver = lambda incident_angle: max(0, 1 - abs(incident_angle))
    gain_facet = lambda incident_angle: max(0, 1 - abs(incident_angle))
    
    sim.add_facet(position=[10,  0, 0], normal=[-1, 0, 0], gain=gain_facet, area=facet_area)
    sim.add_facet(position=[ 5, -1, 0], normal=[0, 0, 1], gain=gain_facet, area=facet_area)
    sim.add_receiver(boresight=[1, 0, 0], gain=gain_receiver, name=receiver_name, area=receiver_area)
    sim.add_pulse(frequency=frequency, power=transmitter_power, phase=0, boresight=[1, 0, 0], gain=gain_antenna)

    sim.run()

    sim

## Assumptions:

The mathematical representation of the polarimetric radar entities and the physical phenomena are limited as follows:
- Electromagnetic wave has no polarization (to be implemented linear and circular polarizations)
- Surfaces modeling is a series of infinitesimal rigid surfaces (aka: [Facet Method](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003RS002903))
- Facets have 1D reflection pattern (to be implemented 2D)
- No receiver equipment (to be implemented monostatic, bi-static, cell-matrix)
- No transmitter equipment (to be implemented)
- Transmitters have 1D radiation pattern (to be implemented 2D)
- No refraction
- No diffraction
- No resonance
- No Doppler effect
- No quantum effect
- No database of real materials (to be implemented)
- No Montecarlo simulation (to be implemented)
- No plots (to be implemented)