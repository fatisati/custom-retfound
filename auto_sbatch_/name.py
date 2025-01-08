def generate_model_name(defaults, params, ABBREVIATIONS):
    job_name_parts = [params['experiment']]
    # Include other parameters that differ from their defaults
    for key, val in params.items():
        if (
            key in ABBREVIATIONS
            and val is not None
            and val != defaults.get(key)
        ):
            
            # Use the abbreviation for the key
            abbreviated_key = ABBREVIATIONS[key]
            # max_length = 4
            shortened_val = str(val) if isinstance(val, str) else str(val)
            job_name_parts.append(f"{abbreviated_key}_{shortened_val}")

    # Join all parts to form the job name
    job_name = "_".join(job_name_parts)
    return job_name