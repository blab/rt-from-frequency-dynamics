struct ModelStan
    model::SampleModel # What type is this
    data::Dict
    posterior::Dict
end

function ModelStan(model::SampleModel, data::Dict)
    return ModelStan(model, data, Dict())
end

function make_stan(model_R::RModel, model_D::ModelData, model_name::String, tmpdir)
    stan_data = make_stan_data(model_R, model_D)
    stan_model = make_stan_model(model_R, model_name, tmpdir)
    return ModelStan(stan_model, stan_data)
end

function load_samples!(MS::ModelStan)
        res, names = read_samples(MS.model, output_format=:array, return_parameters = true)
        MS.posterior["samples"] = res
        MS.posterior["cnames"] = names 
end

function run!(MS::ModelStan; 
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    get_samples = false)

    rc = stan_sample(MS.model; 
        data = MS.data,
        num_warmup = num_warmup,
        num_samples = num_samples,
        num_chains = num_chains) 
    
    if success(rc) & get_samples
        load_samples!(MS)
    end
end
