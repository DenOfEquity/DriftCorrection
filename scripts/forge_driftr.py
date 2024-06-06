import gradio as gr

from modules import scripts
import modules.shared as shared
import torch, math

#effect seems better when aplied to denoised result after CFG, rather than to cond/uncond before CFG

class driftrForge(scripts.Script):
    def __init__(self):
        self.method1 = "None"
        self.method2 = "None"

    def title(self):
        return "Latent Drift Correction"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                method1 = gr.Dropdown(["None", "custom", "mean", "median", "mean/median average", "centered mean", "average of extremes", "average of quantiles"], value="None", type="value", label='Correction method (per channel)')
                method2 = gr.Dropdown(["None", "mean", "median", "mean/median average", "center to quantile", "local average"], value="None", type="value", label='Correction method (overall)')
            with gr.Row():
                strengthC = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=1.0, label='strength (per channel)')
                strengthO = gr.Slider(minimum=-1.0, maximum=1.0, step=0.01, value=0.8, label='strength (overall)')
            with gr.Row(equalHeight=True):
                custom = gr.Textbox(value='0.5 * (M + m)', max_lines=1, label='custom function', visible=True)
                topK = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label='quantiles', visible=False, scale=0)
                blur = gr.Slider(minimum=0, maximum=128, step=1, value=0, label='blur radius (x8)', visible=False, scale=0)
                sigmaWeight = gr.Dropdown(["Hard", "Soft", "None"], value="Hard", type="value", label='Limit effect by sigma', scale=0)
            with gr.Row():
                stepS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label='Start step')
                stepE = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='End step')
            with gr.Row():
                softClampS = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='Soft clamp start step')
                softClampE = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label='Soft clamp end step')

            def show_topK(m1, m2):
                if m1 == "centered mean" or m1 == "average of extremes" or m1 == "average of quantiles":
                    return gr.update(visible=True), gr.update(visible=False)
                elif m2 == "center to quantile":
                    return gr.update(visible=True), gr.update(visible=False)
                elif m2 == "local average":
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=False), gr.update(visible=False)

            method1.change(
                fn=show_topK,
                inputs=[method1, method2],
                outputs=[topK, blur],
                show_progress=False
            )
            method2.change(
                fn=show_topK,
                inputs=[method1, method2],
                outputs=[topK, blur],
                show_progress=False
            )

        self.infotext_fields = [
            (method1,       "ldc_method1"),
            (method2,       "ldc_method2"),
            (topK,          "ldc_topK"),
            (blur,          "ldc_blur"),
            (strengthC,     "ldc_strengthC"),
            (strengthO,     "ldc_strengthO"),
            (stepS,         "ldc_stepS"),
            (stepE,         "ldc_stepE"),
            (sigmaWeight,   "ldc_sigW"),
            (softClampS,    "ldc_softClampS"),
            (softClampE,    "ldc_softClampE"),
            (custom,        "ldc_custom"),
        ]

        return method1, method2, topK, blur, strengthC, strengthO, stepS, stepE, sigmaWeight, softClampS, softClampE, custom


    def patch(self, model):
        model_sampling = model.model.model_sampling
        sigmin = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_min))
        sigmax = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max))


##  https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
        def soft_clamp_tensor(input_tensor, threshold=3.5, boundary=4):
            if max(abs(input_tensor.max()), abs(input_tensor.min())) < 4:
                return input_tensor
            channel_dim = 1

            max_vals = input_tensor.max(channel_dim, keepdim=True)[0]
            max_replace = ((input_tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
            over_mask = (input_tensor > threshold)

            min_vals = input_tensor.min(channel_dim, keepdim=True)[0]
            min_replace = ((input_tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
            under_mask = (input_tensor < -threshold)

            return torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, input_tensor))

        def center_latent_mean_values(latent, multiplier):
            thisStep = shared.state.sampling_step
            lastStep = shared.state.sampling_steps

            channelMultiplier = multiplier * self.strengthC
            fullMultiplier = multiplier * self.strengthO
          
            if thisStep >= self.stepS * lastStep and thisStep <= self.stepE * lastStep:
                for b in range(len(latent)):
                    for c in range(3):
                        custom = None
                        channel = latent[b][c]

                        if self.method1 == "mean":
                            custom = "M"
                            #averageMid = torch.mean(channel)
                            #latent[b][c] -= averageMid  * channelMultiplier

                        elif self.method1 == "median":
                            custom = "m"
                            #averageMid = torch.quantile(channel, 0.5)
                            #latent[b][c] -= averageMid  * channelMultiplier

                        elif self.method1 == "mean/median average":
                            custom = "0.5 * (M+m)"
                            #averageMid = 0.5 * (torch.mean(channel) + torch.quantile(channel, 0.5))
                            #latent[b][c] -= averageMid  * channelMultiplier
                            

                        elif self.method1 == "centered mean":
                            custom="rM(self.topK, 1.0-self.topK)"
##                            valuesHi = torch.topk(channel, int(len(channel)*self.topK), largest=True).values
##                            valuesLo = torch.topk(channel, int(len(channel)*self.topK), largest=False).values
##                            averageMid = torch.mean(channel).item() * len(channel)
##                            averageMid -= torch.mean(valuesHi).item() * len(channel)*self.topK
##                            averageMid -= torch.mean(valuesLo).item() * len(channel)*self.topK
##                            averageMid /= len(channel)*(1.0 - 2*self.topK)
##                            latent[b][c] -= averageMid  * channelMultiplier

                        elif self.method1 == "average of extremes":
                            custom="0.5 * (inner_rL(self.topK) + inner_rH(self.topK))"
##                            valuesHi = torch.topk(channel, int(len(channel)*self.topK), largest=True).values
##                            valuesLo = torch.topk(channel, int(len(channel)*self.topK), largest=False).values
##                            averageMid = 0.5 * (torch.mean(valuesHi).item() + torch.mean(valuesLo).item())
##                            latent[b][c] -= averageMid  * channelMultiplier

                        elif self.method1 == "average of quantiles":
                            custom="0.5 * (q(self.topK) + q(1.0-self.topK))"
##                            averageMid = 0.5 * (torch.quantile(channel, self.topK) + torch.quantile(channel, 1.0 - self.topK))
##                            latent[b][c] -= averageMid  * channelMultiplier

                        elif self.method1 == "custom":
                            custom = self.custom

                        if custom != None:
                            M = torch.mean(channel)
                            m = torch.quantile(channel, 0.5)
                            def q(quant):
                                return torch.quantile(channel, quant)
                            def qa(quant):
                                return torch.quantile(abs(channel), quant)
                            def inner_rL(lo):   #   mean of values from lowest to input(proportional)
                                valuesLo = torch.topk(channel, int(len(channel)*lo), largest=False).values
                                return torch.mean(valuesLo).item()
                            def inner_rH(hi):   #   mean of values from input(proportional) to highest
                                valuesHi = torch.topk(channel, int(len(channel)*hi), largest=True).values
                                return torch.mean(valuesHi).item()
                                
                            def rM(rangelo, rangehi):       #   mean of range
                                if rangelo == rangehi:
                                    return M
                                else:
                                    averageHi = inner_rH(1.0-rangehi)
                                    averageLo = inner_rL(rangelo)

                                    average = torch.mean(channel).item() * len(channel)
                                    average -= averageLo * len(channel) * rangelo
                                    average -= averageHi * len(channel) * (1.0-rangehi)
                                    average /= len(channel)*(rangehi - rangelo)
                                    return average

                            averageMid = eval(custom)
                            latent[b][c] -= averageMid  * channelMultiplier
                            
                    if self.method2 == "mean":
                        latent[b] -= latent[b].mean() * fullMultiplier
                    elif self.method2 == "median":
                        latent[b] -= latent[b].median() * fullMultiplier
                    elif self.method2 == "mean/median average":
                        mm = latent[b].mean() + latent[b].median()
                        latent[b] -= 0.5 * fullMultiplier * mm
                    elif self.method2 == "center to quantile":
                       quantile = torch.quantile(latent[b].flatten(), self.topK) #   0.5 is same as median
                       latent[b] -= quantile * fullMultiplier
                    elif self.method2 == "local average" and fullMultiplier != 0.0 and self.blur != 0:
                        import torchvision.transforms.functional as TF
                        blurred = TF.gaussian_blur(latent[b], 1+self.blur+self.blur)       # blur size as input?
                        torch.lerp(latent[b], blurred, fullMultiplier, out=latent[b])
                        del blurred


            if thisStep >= self.softClampS * lastStep and thisStep <= self.softClampE * lastStep:
                for b in range(len(latent)):
                    latent[b] = soft_clamp_tensor (latent[b])


            return latent


        def map_sigma(sigma, sigmax, sigmin):
            return (sigma - sigmin) / (sigmax - sigmin)

        def center_mean_latent_post_cfg(args):
            denoised = args["denoised"]
            sigma    = args["sigma"][0]

            if self.sigmaWeight == "None":                  #   range 1 - always full correction
                mult = 1
            else:
                mult = map_sigma(sigma, sigmax, sigmin)     #   range 0.0 to 1.0
                if self.sigmaWeight == "Soft":              #   range 0.5 to 1.0
                    mult += 1.0
                    mult /= 2.0

            denoised = center_latent_mean_values(denoised, mult)        
            return denoised

        m = model.clone()
        m.set_model_sampler_post_cfg_function(center_mean_latent_post_cfg)

        return (m, )


    def process_before_every_sampling(self, params, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        method1, method2, topK, blur, strengthC, strengthO, stepS, stepE, sigmaWeight, softClampS, softClampE, custom = script_args

        if method1 == "None" and method2 == "None":
            return

        self.method1 = method1
        self.method2 = method2
        self.topK = topK
        self.blur = blur
        self.strengthC = strengthC
        self.strengthO = strengthO
        self.stepS = stepS
        self.stepE = stepE
        self.sigmaWeight = sigmaWeight
        self.softClampS = softClampS
        self.softClampE = softClampE
        self.custom = custom
        

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        params.extra_generation_params.update(dict(
            ldc_method1 = method1,
            ldc_method2 = method2,
            ldc_strengthC = strengthC,
            ldc_strengthO = strengthO,
            ldc_stepS = stepS,
            ldc_stepE = stepE,
            ldc_sigW = sigmaWeight,
            ldc_softClampS = softClampS,
            ldc_softClampE = softClampE,
        ))
        if method1 == "custom":
            params.extra_generation_params.update(dict(ldc_custom = custom, ))
        if method1 == "centered mean" or method1 == "average of extremes" or method1 == "average of quantiles" or method2 == "center to quantile":
            params.extra_generation_params.update(dict(ldc_topK = topK, ))
        if method2 == "local average":
            params.extra_generation_params.update(dict(ldc_blur = blur, ))

        unet = params.sd_model.forge_objects.unet
        unet = driftrForge.patch(self, unet)[0]
        params.sd_model.forge_objects.unet = unet

        return
