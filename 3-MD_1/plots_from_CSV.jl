
using DataFrames, CSV, Gadfly
#
# pyplot(size=(800, 600))
# PyPlot.PyObject(PyPlot.axes3D)
# fnt = "sans-serif"
# default(titlefont=Plots.font(fnt,24), guidefont=Plots.font(fnt,24), tickfont=Plots.font(fnt,14), legendfont=Plots.font(fnt,14))

Ts = [1.0 1.7 3.0 8.0]
N = 256
DF = Array{DataFrame}(length(Ts))

for i = 1:length(Ts)
    DF[i] = CSV.read(string("./Data/PV_", N,"_T", Ts[i], ".csv"))
end

# test con StatPlots e PyPlot
#@df DF[3] plot(:V, :P, xaxis=("V",(0,2000)), yaxis=("P",(-1,ceil(:P[end]))), linewidth=2, leg=false)

# test con Gadfly
light_panel = Theme(
    panel_fill="white",
    default_color="blue"
)

ticks = [0 1e3 2e3 3e3]

Gadfly.push_theme(:default)
l1 = layer(DF[1], x=:V, y=:P, ymin=:dP, ymax=:dP, Geom.line, Geom.ribbon)
l2 = layer(DF[2], x=:V, y=:P, ymin=:dP, ymax=:dP, Geom.line, Geom.ribbon)
l3 = layer(DF[3], x=:V, y=:P, ymin=:dP, ymax=:dP, Geom.line, Geom.ribbon)
l4 = layer(DF[4], x=:V, y=:P, ymin=:dP, ymax=:dP, Geom.line, Geom.ribbon)

PV = Gadfly.plot(l1, l2, l3, l4, Guide.xticks(ticks=ticks),
style(major_label_font="CMU Serif",minor_label_font="CMU Serif"))
Gadfly.pop_theme()

#p1 = Gadfly.plot(DF[2], x=:V, y=:P, ymin=:dP, ymax=:dP, Geom.line, Geom.ribbon, style(major_label_font="CMU Serif",minor_label_font="CMU Serif"))
file = string("./Plots/PVG_", N, ".pdf")
draw(PDF(file, 8cm, 6cm), PV)
