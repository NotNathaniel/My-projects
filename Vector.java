//Write class vector
public class Vector{
  double x;
  double y;
  
  
  // Constructs vector with specified coordinates
  //ideally between 0 and 1

  public Vector(double x, double y){
    this.x=x;
    this.y=y;
  }
  
  //adds 'original'vector with input vector
  public Vector add(Vector v){
    return new Vector(x+v.x, y+v.y);
    
  }
  //computes angle of vector (w.r.t. y-axis?)
  public double angle(){
   return Math.atan2(y,x);
  }
  
  //returns distance between original vector and some other vector
  public double distance(Vector v){
   return Math.sqrt((x-v.x)*(x-v.x)+(y-v.y)*(y-v.y));//just sqrt of sum of square differences 
  }
  
  
  //computes scalar product between input vector and original vector
  public double dot(Vector v){
    
    return x*v.x+y*v.y;
  }
  
  //flips sign of x element
  public Vector flipSignX(){
    //but you want to return a vector, not just change the array, something is clearly wrong here..
   return new Vector(-x,y);
  }
  
  
  
  //flips sign of y element
  public Vector flipSignY(){
    return new Vector(x,-y);
  }
  
  //return x element
  public double getX(){
   return x;//how would you do something like this? 
  }
  //return y element
  public double getY(){
   return y;
  }
  //computes lenght of vector  
  public double length(){
   return Math.sqrt(x*x+y*y);
  }
  
  //creates vector object using polar coordinates
  public static Vector polar(double length, double angle){
    return new Vector(length*Math.cos(angle),length*Math.sin(angle));
    
  }
  
  //creates a random vector of lenght len
  public static Vector randomVector(double len){
    Vector vec=polar(len,2*Math.PI*Math.random());
    return vec;
    
  }
 //scales the original vector by d
  public Vector scale(double d){
   return new Vector(d*x,d*y);
  }
  
  //subtracts v from the original vector 
  public Vector sub(Vector v){
    return new Vector(x-v.x,y-v.y); 
  }
  //text representation of the vector
  public String toString(){
   String str=Double.toString(x)+"\n";
   str=str+Double.toString(y)+"\n";
   return str; 
  }
  
  
  //testing
  
  public static void main(String[] args){
    Vector vector=new Vector(0.5,0.5);
    Vector vec2=new Vector(0.4,0.4);
    System.out.println(vector.toString());
    System.out.println(vec2.toString());
    //add
    System.out.println("Add "+vector.add(vec2));//right

    //angle
    System.out.println("Angle "+vector.angle());//right
    //distance
    System.out.println("Distance "+vector.distance(vec2));//right
    //dot
    System.out.println("Dot "+vector.dot(vec2));//right
    //flipsignX
    
    System.out.println("FlipsignX "+vector.flipSignX());//wrong
    //flipsign y
    System.out.println("FlipsignY "+vector.flipSignY() );//wrogn
    //getX
    System.out.println("X "+vector.getX());
    //getY
    System.out.println("Y "+vector.getY());
    //length
    System.out.println("length "+vector.length());
    //polar
    Vector pol=polar(1,Math.PI);
    System.out.println("polar "+pol);
    System.out.println("length and angle of polar"+pol.length()+pol.angle());
    //randomVector
    System.out.println("randomVector"+randomVector(1));
    System.out.println("its length"+randomVector(1).length() );
    //scale
    System.out.println("scale "+vector.scale(2));
    //sub
    System.out.println("Subtract "+vector.sub(vec2));
   
  }


}


