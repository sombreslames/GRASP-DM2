<?php
// src/AppBundle/Entity/User.php
namespace AppBundle\Entity;

use Symfony\Component\Security\Core\User\UserInterface;
use Symfony\Component\Validator\Constraints as Assert;
use Symfony\Bridge\Doctrine\Validator\Constraints\UniqueEntity;
use Doctrine\ORM\Mapping as ORM;
use AppBundle\Repository\UserRepository;
/**
 * @ORM\Entity
 * @ORM\Table(name="User")
 * @UniqueEntity(fields="email",  message="It looks like your already have an account!")
 * @UniqueEntity(fields="username",  message="Username is already in use, sorry bro :( !")
 * @ORM\Entity(repositoryClass="AppBundle\Repository\UserRepository")
 */
class User implements UserInterface
{
    /**
     * @ORM\Id
     * @ORM\GeneratedValue(strategy="AUTO")
     * @ORM\Column(name="ID_user", type="integer")
     */
    private $ID_user;

    /**
     * @Assert\NotBlank(groups={"Registration"})
     * @ORM\Column(type="string", length=255, unique=true)
     */
    private $username;

    /**
     * @Assert\NotBlank(groups={"Registration"})
     * @Assert\Email()
     * @ORM\Column(type="string", length=255, unique=true)
     */
    private $email;

    /**
     * @ORM\Column(type="string", length=64)
     */
    private $password;

    /**
     * @Assert\NotBlank(groups={"Registration"})
     * @Assert\Length(max=4096)
     */
    private $plainPassword;


    /**
     * @ORM\Column(type="json_array")
     */
    private $roles = array();

    /**
     * @ORM\Column(type="boolean")
     */
    private $isActive;

    public function getID_user(){
        return $this->ID_user;
    }
    public function getPlainPassword()
    {
        return $this->plainPassword;
    }

    public function setPlainPassword($plainPassword)
    {
        $this->plainPassword = $plainPassword;

        $this->password = null;
    }

    public function setEmail($email)
    {
        $this->email=$email;
    }
    public function getEmail(){
        return $this->email;
    }



    public function getUsername()
    {
        return $this->username;
    }

    public function setUsername($username)
    {
        $this->username=$username;
    }

    public function getRoles()
    {
        return ['ROLE_USER'];
    }

    /**
     * @param mixed $password
     */
    public function setPassword($password)
    {
        $this->password = $password;
    }
    public function getPassword()
    {
        return $this->password;
    }
    /**
     * @param mixed $isActive
     */
    public function setIsActive($isActive)
    {
        $this->isActive = $isActive;
    }

    public function getIsActive(){
        return $this->isActive;
    }
    public function getSalt()
    {
    }

    public function eraseCredentials()
    {
        $this->plainPassword=null;
    }

// more getters/setters
}