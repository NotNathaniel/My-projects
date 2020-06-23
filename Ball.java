import java.awt.*;
public class Ball{
  
  //should all preferably be private.
  private Vector position;
  private Vector velocity;
  private Color c;
  private double size;
  private Box box;
  
  public Ball(Vector position,Vector velocity,Color c,double size, Box box){
   this.position=position;
   this.velocity=velocity;
   this.c=c;
   this.size=size;
   this.box=box;
 }

//I decided to just immediately do these things in the Ball function
   public void setPosition(Vector pos){
   this.position=pos;  
  }
  
  public void setVelocity(Vector vel){
    //should you this.pos= what will be on the rhs?
    this.velocity=velocity;
   }

  public void setColor(Color c){

    this.c=c;
   }

  public void setSize(int size){

    this.size=size;
   }
  
  public Vector getPosition(){
    return this.position;
  }
  
  public Vector getVelocity(){
   return this.velocity; 
  }
    public double getSize(){
    return this.size;
  }
    
    
      public void addSize(double c){
    this.size+=c;
}
  
  
  
  public void move(double step) {
         //moving
         position=position.add(velocity.scale(step));
         //gravity
         double xdir=Math.cos(Math.PI*Math.round(Math.random()));//Math.sign doesn't exist
         //ball bounces in random directions but gravity exists
         velocity=velocity.add(new Vector(xdir/20,0.05));
           
           
         //collision with floor:
         if (position.y > box.getHeight()-size && velocity.y>0) {
            velocity=velocity.flipSignY();
            velocity=velocity.scale(0.95);
            position.y=box.getHeight()-size;
       
            //velocity.y = velocity.y - box.getHeight()/10;        // Friction in the bounce
       }

       // Collison with the roof
       if (position.y<size && velocity.y<0) {
           velocity = velocity.flipSignY().scale(0.8);
           
       }

       // Collision with walls
       if (position.x < size && velocity.x < 0 ||              // left wall
           position.x > box.getWidth()-size && velocity.x > 0) { // right wall
           velocity.x = -velocity.x*0.9;
           
       }
       
       //collision with other balls
       for(Ball balls: box.getBalls()){
         this.collision(balls);
    }
   }
        
  public boolean collision(Ball a){
    //to end the function
    if(a==this)
      return false;
    else if(position.distance(a.getPosition())<size+a.getSize()){
      collide(a);
      return true;
    }
    else
      return false;
  }
  
  // bigger ball cuts smaller ball
  void collide(Ball b){
    if(size>b.getSize()){
      
      b.velocity=b.velocity.scale(1.01);//RUN!
      
      b.addSize(-b.getSize()/50);
      if(b.getSize()<=0){
        box.getBalls().remove(b);
      }
    }
  }

  
  public void paint(Graphics g) {
       g.setColor(c);
       int x1=(int)(position.x-size);
       int y1=(int)(position.y-size);
       int s=2*(int)size;
       g.fillOval(x1, y1, s,s);//x-size to 2*size
    }
 
  
  public String toString(){
   String str="Position "+position.toString()+"\n";
   str=str+"Velocity "+velocity.toString()+"\n";
   str=str+ "Color "+c.toString()+"\n";
   str=str+"Size"+Double.toString(size);
   
     
     
   return str; 
  }
  
   
}
  
