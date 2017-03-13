
data = load("data.txt");
x = data(:,1);
y = data(:,2);

printf("the number of points is %d\n", rows(x))
count=0;
for i = 1:rows(x)
  if (x(i) < 0 && y(i) < 0)
    count+=1;
  endif
endfor
per = ((count/rows(x))*100);
printf("the percentage of negative is %d\n", per)


function plotData (x,y)
  #pontos
  figure(1)
  plot(x,y, "xr")
  xlabel("x")
  ylabel("y")
  title("file data.txt")
  legend("data points")
  #histogramas
  figure(2)
  subplot(2,1,2)
  hist(x)
  title("histogram of X")
  subplot(2,1,1)
  hist(y)
  title("histogram of Y")
endfunction

function ret = normalize(x)
  normX  = x - min(x(:));
  ret = normX ./ max(normX(:));
endfunction

#1
#plotData(x,y)

#2
%{
err = 2;
n = 0;
while(err > 1)
  n+=1;
  p  = polyfit(x,y,n);
  fx = polyval(p,x);
  err(n) = mean(power((y-fx),2))
  plot(err)
endwhile
%}

#3


normx = normalize(x);
normy = normalize(y);
#plotData(normx,normy)

#apartir daqui esta mal ate ao fim do ex
err = 2;
n = 30;
for i=1:n
  p  = polyfit(normx,normy,1);
  fx = polyval(p,normx);
  err(i) = mean(power((normy-fx),2))
  plot(err)
endfor
#ate aqui


#5
%{
flippedx = flipud(x);
meanx = [];
for i=1:rows(x)
  meanx = [meanx; ((x(i)+flippedx(i))/2)];
endfor
matrix = [x, flippedx, meanx]
%}


#6
