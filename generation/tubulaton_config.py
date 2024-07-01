default_config_dict = {
    "num_input_vtk": "2",

    #Make sure these are equal to avoid wasted vtk savefiles (they are just deleted!)
    "vtk_steps": "5",
    "nb_max_steps": "5",

    #Any positive number
    "Angle_bundle": "0.7",
    "Angle_cut": "0.6",
    "Angle_mb_limite": "0.7",
    "Angle_mb_trajectoire": "0.7",

    
    #Any natural number
    "D_bundle": "49",
    "nb_microtubules_init": "30",
    "nb_microtubules_init_2": "30",

    #Any probability
    "proba_crossmicro_cut": "0.005",
    "proba_crossmicro_shrink": "0.001",
    "proba_detachement_par_step_par_microtubule": "0.001",
    "proba_initialisation_area": "0.002",
    "proba_initialisation_area_2": "0.002",
    "proba_shrink": "0.0001",

    "nom_folder_input_vtk": "",
    "nom_folder_output_vtk": "",

    "Nucleation_Direction": "0",
    "Nucleation_Direction_2": "1",
    "Output_Contour": "0",
    "Output_Contour_2": "0",
    "cortical": "0",
    "cutgroup": "0",
    "d_mb": "10",
    "decision_accrochage": "0",
    "decision_cut": "0",
    "decision_rencontre": "-1",
    "details": "1",
    "epaisseur_corticale": "1.0",
    "garbage_steps": "500",
    "gr_st_moins": "0",
    "gr_st_plus": "1",
    "nom_config": "config.ini",
    "nom_input_vtk": "",
    "nom_input_vtk_2": "",
    "nom_input_vtk_ref": "",
    "nom_input_vtk_ref_2": "",
    "nom_output_vtk": "NucleusInsideCyl_",
    "nom_rapport": "NucleusInsideCyl.txt",
    "part_alea_alea": "1",
    "part_alea_fixe": "39",
    "part_influence_influence": "0",
    "part_influence_normale": "1",
    "proba_rescue": "0",
    "proba_tocut": "0",
    "save_events": "10000",
    "stop": "0",
    "taille_microtubule": "8.0",
    "tan_Angle_mb": "0.839099631177"
}
