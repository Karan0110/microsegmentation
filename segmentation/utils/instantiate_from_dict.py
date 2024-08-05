from typing import Any

def instantiate_from_dict(namespace, 
                          information : dict) -> Any:
    if 'name' not in information:
        raise ValueError(f"To instantiate object from dict a class name must be provided!")
    class_name = information['name']

    if not hasattr(namespace, class_name):
        raise ValueError(f"{namespace} does not contain the class: {class_name}")
    Class = getattr(namespace, class_name)

    params = information.get('params', {})

    instantiated_object = Class(**params)

    return instantiated_object
